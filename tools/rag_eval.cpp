#include <rag/engine.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
#include <string>

int main(int argc, char** argv) {
    try {
        const std::string path =
            argc > 1 ? argv[1] : std::string(LRS_SOURCE_DIR) + "/tests/fixtures/qrels.json";
        std::ifstream input(path);
        if (!input)
            throw std::runtime_error("cannot open " + path);
        nlohmann::json fixture;
        input >> fixture;

        rag::Engine engine;
        for (const auto& doc : fixture.at("documents")) {
            auto added = engine.add(doc.at("id"), doc.at("text"), {}, doc.value("title", ""));
            if (!added)
                throw std::runtime_error(added.error().message);
        }
        if (auto built = engine.build(); !built)
            throw std::runtime_error(built.error().message);

        nlohmann::json report;
        for (const auto profile :
             {rag::retrieval::Profile::efficiency, rag::retrieval::Profile::balanced,
              rag::retrieval::Profile::quality}) {
            double recall = 0.0, mrr = 0.0, ndcg = 0.0;
            for (const auto& query : fixture.at("queries")) {
                const std::set<std::string> relevant(query.at("relevant").begin(),
                                                     query.at("relevant").end());
                rag::retrieval::SearchOptions options;
                options.profile = profile;
                options.top_k = fixture.value("k", 5U);
                const auto results =
                    engine.search(query.at("text").get<std::string>(), options, nullptr);
                if (!results)
                    throw std::runtime_error(results.error().message);
                std::size_t found = 0;
                double dcg = 0.0;
                double reciprocal_rank = 0.0;
                for (std::size_t i = 0; i < results->size(); ++i) {
                    if (!relevant.contains((*results)[i].uri))
                        continue;
                    ++found;
                    if (reciprocal_rank == 0.0)
                        reciprocal_rank = 1.0 / static_cast<double>(i + 1);
                    dcg += 1.0 / std::log2(static_cast<double>(i + 2));
                }
                recall += relevant.empty()
                              ? 1.0
                              : static_cast<double>(found) / static_cast<double>(relevant.size());
                mrr += reciprocal_rank;
                double ideal = 0.0;
                for (std::size_t i = 0; i < std::min(relevant.size(), options.top_k); ++i)
                    ideal += 1.0 / std::log2(static_cast<double>(i + 2));
                ndcg += ideal == 0.0 ? 1.0 : dcg / ideal;
            }
            const double count = static_cast<double>(fixture.at("queries").size());
            report[rag::retrieval::name(profile)] = {
                {"recall_at_k", recall / count}, {"mrr", mrr / count}, {"ndcg_at_k", ndcg / count}};
        }
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "lrs-rag-eval: " << error.what() << '\n';
        return 1;
    }
}
