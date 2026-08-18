#include "qrels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <set>

#include <nlohmann/json.hpp>

#include <rag/backend/embedded_backend.hpp>
#include <rag/dense/backends.hpp>
#include <rag/dense/simd.hpp>
#include <rag/preparation/document_preparer.hpp>
#include <rag/retrieval/runtime.hpp>

#include "common.hpp"

namespace lrs::eval {
namespace {

std::size_t candidate_pool(std::string_view profile, std::size_t top_k) {
    if (profile == "efficiency")
        return std::max(3 * top_k, std::size_t{24});
    if (profile == "quality")
        return std::max(20 * top_k, std::size_t{200});
    return std::max(6 * top_k, std::size_t{60});
}

} // namespace

rag::Result<nlohmann::json> run_qrels(const std::string& path) {
    try {
        std::ifstream input(path);
        if (!input)
            return rag::fail<nlohmann::json>(rag::Errc::io_error,
                                             "cannot open qrels fixture: " + path);
        nlohmann::json fixture;
        input >> fixture;
        const std::size_t dimension = fixture.value("dimension", 384U);
        const rag::dense::AnyEmbedder embedder{rag::dense::HashEmbedder{dimension}};
        rag::preparation::PrepareOptions preparation;
        preparation.chunking.max_lines = fixture.value("max_lines", 40U);
        preparation.chunking.overlap_lines = 0;
        rag::backend::EmbeddedMaintenancePolicy policy;
        policy.automatic_compaction = false;
        policy.dense.algorithm = rag::dense::DenseAlgorithm::exact;
        rag::backend::EmbeddedBackend backend(policy);

        std::vector<double> update_to_search_ms;
        std::uint64_t revision = 1;
        for (const auto& value : fixture.at("documents")) {
            const auto started = std::chrono::steady_clock::now();
            auto document = rag::preparation::prepare_document(
                value.at("id"), value.at("text"), value.value("metadata", rag::Metadata{}),
                value.value("title", ""), preparation, &embedder);
            if (!document)
                return rag::unexpected(document.error());
            const auto query_vector = document->chunks.front().embedding;
            if (auto activated = backend.activate(std::move(*document), revision++); !activated)
                return rag::unexpected(activated.error());
            auto visible = backend.dense_candidates({query_vector, 1, {}});
            if (!visible || visible->empty())
                return rag::fail<nlohmann::json>(rag::Errc::corrupt_index,
                                                 "activated document was not immediately visible");
            update_to_search_ms.push_back(std::chrono::duration<double, std::milli>(
                                              std::chrono::steady_clock::now() - started)
                                              .count());
        }
        const auto build_started = std::chrono::steady_clock::now();
        if (auto compacted = backend.compact(); !compacted)
            return rag::unexpected(compacted.error());
        const double build_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - build_started)
                                    .count();

        nlohmann::json report;
        const std::size_t top_k = fixture.value("k", 5U);
        for (const std::string profile : {"efficiency", "balanced", "quality"}) {
            double recall = 0.0;
            double mrr = 0.0;
            double ndcg = 0.0;
            double filtered_recall = 0.0;
            std::size_t filtered_queries = 0;
            std::vector<double> latencies_ms;
            for (const auto& query : fixture.at("queries")) {
                const std::set<std::string> relevant(query.at("relevant").begin(),
                                                     query.at("relevant").end());
                auto vector = embedder.embed_one(query.at("text").get<std::string>());
                if (!vector)
                    return rag::unexpected(vector.error());
                rag::dense::normalize(*vector);
                rag::backend::SearchRequest request;
                request.query = query.at("text").get<std::string>();
                request.embedding = std::move(*vector);
                request.mode = rag::backend::SearchMode::hybrid;
                request.top_k = top_k;
                request.candidate_pool = candidate_pool(profile, top_k);
                if (query.contains("filter"))
                    request.filter =
                        rag::backend::MetadataFilter(query.at("filter").get<rag::Metadata>());
                request.profile = profile;
                const auto query_started = std::chrono::steady_clock::now();
                auto results = rag::retrieval::search(backend, request);
                latencies_ms.push_back(std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() - query_started)
                                           .count());
                if (!results)
                    return rag::unexpected(results.error());
                std::size_t found = 0;
                double dcg = 0.0;
                double reciprocal_rank = 0.0;
                for (std::size_t index = 0; index < results->size(); ++index) {
                    if (!relevant.contains((*results)[index].document_key))
                        continue;
                    ++found;
                    if (reciprocal_rank == 0.0)
                        reciprocal_rank = 1.0 / static_cast<double>(index + 1);
                    dcg += 1.0 / std::log2(static_cast<double>(index + 2));
                }
                const double query_recall =
                    relevant.empty()
                        ? 1.0
                        : static_cast<double>(found) / static_cast<double>(relevant.size());
                recall += query_recall;
                if (!request.filter.empty()) {
                    filtered_recall += query_recall;
                    ++filtered_queries;
                }
                mrr += reciprocal_rank;
                double ideal = 0.0;
                for (std::size_t index = 0; index < std::min(relevant.size(), top_k); ++index)
                    ideal += 1.0 / std::log2(static_cast<double>(index + 2));
                ndcg += ideal == 0.0 ? 1.0 : dcg / ideal;
            }
            const double count = static_cast<double>(fixture.at("queries").size());
            report[profile] = {
                {"recall_at_k", recall / count},
                {"mrr", mrr / count},
                {"ndcg_at_k", ndcg / count},
                {"filtered_recall_at_k",
                 filtered_queries == 0
                     ? nlohmann::json(nullptr)
                     : nlohmann::json(filtered_recall / static_cast<double>(filtered_queries))},
                {"query_latency_ms",
                 {{"p50", percentile(latencies_ms, 0.50)},
                  {"p95", percentile(latencies_ms, 0.95)}}}};
        }
        auto stats = backend.stats();
        if (!stats)
            return rag::unexpected(stats.error());
        report["object"] = "rag.evaluation_report";
        report["kind"] = "qrels";
        report["build_ms"] = build_ms;
        report["update_to_search_latency_ms"] = {{"p50", percentile(update_to_search_ms, 0.50)},
                                                 {"p95", percentile(update_to_search_ms, 0.95)}};
        report["dense_memory_bytes"] = {{"resident", stats->dense_bytes},
                                        {"mapped", stats->dense_mapped_bytes},
                                        {"catalog_embeddings", stats->catalog_embedding_bytes}};
        report["process_rss_bytes"] = process_rss_bytes();
        report["documents"] = stats->live_documents;
        report["chunks"] = stats->live_chunks;
        report["dimension"] = stats->embedding_dimension;
        return report;
    } catch (const std::exception& error) {
        return rag::fail<nlohmann::json>(rag::Errc::parse_error,
                                         "qrels evaluation failed: " + std::string(error.what()));
    }
}

} // namespace lrs::eval
