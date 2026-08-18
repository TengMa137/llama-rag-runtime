#include "dense.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include <rag/dense/index.hpp>
#include <rag/dense/simd.hpp>

#include "common.hpp"
#include "corpus.hpp"

namespace lrs::eval {
namespace {

double required_recall(const DenseOptions& options) {
    using rag::dense::DenseAlgorithm;
    switch (options.policy.algorithm) {
        case DenseAlgorithm::exact:
        case DenseAlgorithm::flat:
            return 1.0;
        case DenseAlgorithm::hnsw:
            return 0.95;
        case DenseAlgorithm::ivf_sq8:
            return 0.93;
        case DenseAlgorithm::ivf_pq:
            return 0.90;
        default:
            return 0.0;
    }
}

} // namespace

rag::Result<nlohmann::json> run_dense(const DenseOptions& options) {
    if (options.vectors == 0 || options.dimension == 0 || options.queries == 0 || options.k == 0 ||
        options.k > options.vectors)
        return rag::fail<nlohmann::json>(rag::Errc::invalid_argument,
                                         "dense evaluation options are invalid");
    if (options.enforce_gate &&
        (options.vectors != 100'000 || options.dimension != 384 || options.k != 10))
        return rag::fail<nlohmann::json>(
            rag::Errc::invalid_argument,
            "quality gates require exactly 100000 vectors, 384 dimensions, and k=10");
    try {
        auto corpus = generate_corpus(options.vectors, options.dimension, options.seed);
        auto rows = corpus.records();
        std::uint64_t state = options.seed;

        rag::dense::NativeExactIndex oracle;
        const auto oracle_started = std::chrono::steady_clock::now();
        if (auto built = oracle.build(rows); !built)
            return rag::unexpected(built.error());
        const double oracle_build_ms = std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() - oracle_started)
                                           .count();

        const auto build_started = std::chrono::steady_clock::now();
        auto target = rag::dense::build_dense_index(options.policy, rows);
        if (!target)
            return rag::unexpected(target.error());
        const double build_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - build_started)
                                    .count();

        double nearest_recall = 0.0;
        double overlap_recall = 0.0;
        double filtered_recall = 0.0;
        std::vector<double> latencies;
        latencies.reserve(options.queries);
        for (std::size_t query_index = 0; query_index < options.queries; ++query_index) {
            const std::size_t selected =
                static_cast<std::size_t>(random_u64(state) % options.vectors);
            rag::Vector query = perturbed_query(corpus.vectors[selected], state);
            auto expected = oracle.search(query, {}, options.k);
            if (!expected)
                return rag::unexpected(expected.error());
            const auto query_started = std::chrono::steady_clock::now();
            auto actual = (*target)->search(query, {}, options.k);
            latencies.push_back(std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - query_started)
                                    .count());
            if (!actual)
                return rag::unexpected(actual.error());
            std::unordered_set<rag::backend::ChunkKey> actual_keys;
            for (const auto& candidate : *actual)
                actual_keys.insert(candidate.chunk);
            if (!expected->empty() && actual_keys.contains(expected->front().chunk))
                nearest_recall += 1.0;
            std::size_t overlap = 0;
            for (const auto& candidate : *expected)
                overlap += actual_keys.contains(candidate.chunk) ? 1U : 0U;
            overlap_recall += static_cast<double>(overlap) / expected->size();

            std::vector<rag::backend::ChunkKey> allowed;
            allowed.reserve(std::min<std::size_t>(options.vectors, 101));
            allowed.push_back(corpus.keys[selected]);
            for (std::size_t offset = 1; offset < 101 && allowed.size() < options.vectors; ++offset)
                allowed.push_back(corpus.keys[(selected + offset * 7919) % options.vectors]);
            std::sort(allowed.begin(), allowed.end());
            allowed.erase(std::unique(allowed.begin(), allowed.end()), allowed.end());
            auto filtered_expected = oracle.search(query, {allowed}, options.k);
            auto filtered_actual = (*target)->search(query, {allowed}, options.k);
            if (!filtered_expected)
                return rag::unexpected(filtered_expected.error());
            if (!filtered_actual)
                return rag::unexpected(filtered_actual.error());
            if (filtered_actual->size() != std::min(options.k, allowed.size()))
                return rag::fail<nlohmann::json>(
                    rag::Errc::corrupt_index,
                    "dense evaluation returned an incomplete filtered top-k");
            const auto relevant =
                filtered_expected->empty() ? std::string{} : filtered_expected->front().chunk;
            if (std::any_of(filtered_actual->begin(), filtered_actual->end(),
                            [&](const auto& candidate) { return candidate.chunk == relevant; }))
                filtered_recall += 1.0;
        }
        nearest_recall /= static_cast<double>(options.queries);
        overlap_recall /= static_cast<double>(options.queries);
        filtered_recall /= static_cast<double>(options.queries);
        const double gate = required_recall(options);
        const bool exact_policy = options.policy.algorithm == rag::dense::DenseAlgorithm::exact ||
                                  options.policy.algorithm == rag::dense::DenseAlgorithm::flat;
        if (options.enforce_gate && (nearest_recall + 1.0e-12 < gate || filtered_recall < 1.0 ||
                                     (exact_policy && overlap_recall < 1.0)))
            return rag::fail<nlohmann::json>(rag::Errc::corrupt_index,
                                             "dense recall quality gate failed");

        const auto stats = (*target)->stats();
        return nlohmann::json{
            {"object", "rag.evaluation_report"},
            {"kind", "dense"},
            {"generator", "xorshift64star-unit-v1"},
            {"seed", options.seed},
            {"vectors", options.vectors},
            {"dimension", options.dimension},
            {"queries", options.queries},
            {"k", options.k},
            {"implementation", stats.implementation},
            {"algorithm", stats.algorithm},
            {"exact", stats.exact},
            {"recall_at_k", nearest_recall},
            {"overlap_at_k", overlap_recall},
            {"filtered_recall_at_k", filtered_recall},
            {"required_recall_at_k", gate},
            {"gate_enforced", options.enforce_gate},
            {"build_ms", build_ms},
            {"oracle_build_ms", oracle_build_ms},
            {"query_latency_ms",
             {{"p50", percentile(latencies, 0.50)}, {"p95", percentile(latencies, 0.95)}}},
            {"dense_memory_bytes",
             {{"resident", stats.resident_bytes},
              {"primary_vectors", stats.primary_vector_bytes},
              {"compressed_vectors", stats.compressed_vector_bytes}}},
            {"process_rss_bytes", process_rss_bytes()}};
    } catch (const std::bad_alloc&) {
        return rag::fail<nlohmann::json>(rag::Errc::unavailable,
                                         "dense evaluation exceeded available memory");
    } catch (const std::exception& error) {
        return rag::fail<nlohmann::json>(rag::Errc::invalid_argument,
                                         "dense evaluation failed: " + std::string(error.what()));
    }
}

nlohmann::json corpus_manifest() {
    return {{"object", "rag.evaluation_corpus_manifest"},
            {"version", 1},
            {"generator", "xorshift64star-unit-v1"},
            {"seed", 0x6C72736576616C31ULL},
            {"dimension", 384},
            {"sizes", {25'000, 100'000, 500'000}},
            {"key_format", "chk_eval_<zero-padded-row>"},
            {"normalization", "unit-l2-f32"}};
}

} // namespace lrs::eval
