#include "runtime.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <rag/dense/tiered_index.hpp>

#include "common.hpp"
#include "corpus.hpp"

namespace lrs::eval {
namespace {

struct SearchMeasurement {
    double p50_ms = 0.0;
    double p95_ms = 0.0;
};

rag::Result<SearchMeasurement> measure(const rag::dense::TieredDenseIndex& index,
                                       const CorpusData& corpus, std::size_t base_count,
                                       std::size_t delta_count, std::size_t queries, std::size_t k,
                                       std::uint64_t seed) {
    const std::span<const rag::backend::ChunkKey> base(corpus.keys.data(), base_count);
    const std::span<const rag::backend::ChunkKey> delta(corpus.keys.data() + base_count,
                                                        delta_count);
    std::vector<double> latency;
    latency.reserve(queries);
    for (std::size_t index_query = 0; index_query < queries; ++index_query) {
        const std::size_t selected =
            static_cast<std::size_t>(random_u64(seed) % (base_count + delta_count));
        const auto query = perturbed_query(corpus.vectors[selected], seed);
        const auto started = std::chrono::steady_clock::now();
        auto result = index.search(query, base, delta, k);
        latency.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                .count());
        if (!result || result->size() != k)
            return rag::fail<SearchMeasurement>(rag::Errc::corrupt_index,
                                                "tiered evaluation returned incomplete top-k");
    }
    return SearchMeasurement{percentile(latency, 0.50), percentile(latency, 0.95)};
}

} // namespace

rag::Result<nlohmann::json> run_runtime(const RuntimeOptions& options) {
    if (options.base_vectors == 0 || options.dimension == 0 || options.queries == 0 ||
        options.k == 0 || options.k > options.base_vectors)
        return rag::fail<nlohmann::json>(rag::Errc::invalid_argument,
                                         "runtime evaluation options are invalid");
    try {
        const std::size_t maximum_delta = options.base_vectors / 10;
        auto corpus =
            generate_corpus(options.base_vectors + maximum_delta, options.dimension, options.seed);
        auto rows = corpus.records();
        auto empty = rag::dense::TieredDenseIndex::empty();
        if (!empty)
            return rag::unexpected(empty.error());
        const auto base_rows =
            std::span<const rag::dense::VectorRecord>(rows.data(), options.base_vectors);
        auto base = (*empty)->compact(base_rows, options.dimension, options.policy);
        if (!base)
            return rag::unexpected(base.error());

        nlohmann::json levels = nlohmann::json::array();
        std::shared_ptr<const rag::dense::TieredDenseIndex> ten_percent;
        double baseline_p50 = 0.0;
        for (const std::size_t percent : {0U, 5U, 10U}) {
            const std::size_t delta_count = options.base_vectors * percent / 100;
            const auto delta_rows = std::span<const rag::dense::VectorRecord>(
                rows.data() + options.base_vectors, delta_count);
            const auto update_started = std::chrono::steady_clock::now();
            auto tiered = (*base)->with_delta(delta_rows, {}, options.dimension);
            if (!tiered)
                return rag::unexpected(tiered.error());
            auto first = measure(**tiered, corpus, options.base_vectors, delta_count, 1, options.k,
                                 options.seed ^ percent);
            if (!first)
                return rag::unexpected(first.error());
            const double update_to_search_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                          update_started)
                    .count();
            auto search = measure(**tiered, corpus, options.base_vectors, delta_count,
                                  options.queries, options.k, options.seed ^ (percent + 17));
            if (!search)
                return rag::unexpected(search.error());
            if (percent == 0)
                baseline_p50 = search->p50_ms;
            levels.push_back(
                {{"delta_percent", percent},
                 {"delta_vectors", delta_count},
                 {"query_latency_ms", {{"p50", search->p50_ms}, {"p95", search->p95_ms}}},
                 {"p50_overhead_ratio", baseline_p50 == 0.0 ? 0.0 : search->p50_ms / baseline_p50},
                 {"update_to_search_ms", update_to_search_ms},
                 {"dense_memory_bytes", (*tiered)->stats().resident_bytes}});
            if (percent == 10)
                ten_percent = std::move(*tiered);
        }

        std::atomic<bool> done{false};
        rag::Result<std::shared_ptr<const rag::dense::TieredDenseIndex>> compacted =
            rag::fail<std::shared_ptr<const rag::dense::TieredDenseIndex>>(
                rag::Errc::unavailable, "compaction did not run");
        double compaction_ms = 0.0;
        const auto all_rows = std::span<const rag::dense::VectorRecord>(
            rows.data(), options.base_vectors + maximum_delta);
        std::thread worker([&] {
            const auto started = std::chrono::steady_clock::now();
            compacted = ten_percent->compact(all_rows, options.dimension, options.policy);
            compaction_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - started)
                                .count();
            done.store(true, std::memory_order_release);
        });
        std::size_t searches_during_compaction = 0;
        std::size_t search_failures = 0;
        std::uint64_t peak_rss = process_rss_bytes();
        do {
            auto result = measure(*ten_percent, corpus, options.base_vectors, maximum_delta, 1,
                                  options.k, options.seed + searches_during_compaction);
            ++searches_during_compaction;
            if (!result)
                ++search_failures;
            peak_rss = std::max(peak_rss, process_rss_bytes());
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (!done.load(std::memory_order_acquire));
        worker.join();
        peak_rss = std::max(peak_rss, process_rss_bytes());
        if (!compacted)
            return rag::unexpected(compacted.error());
        if (search_failures != 0 || searches_during_compaction == 0)
            return rag::fail<nlohmann::json>(
                rag::Errc::corrupt_index,
                "queries were not continuously available during compaction");
        const auto final_stats = (*compacted)->stats();
        return nlohmann::json{{"object", "rag.evaluation_report"},
                              {"kind", "runtime"},
                              {"generator", "xorshift64star-unit-v1"},
                              {"seed", options.seed},
                              {"base_vectors", options.base_vectors},
                              {"dimension", options.dimension},
                              {"queries_per_level", options.queries},
                              {"k", options.k},
                              {"implementation", final_stats.base_implementation},
                              {"algorithm", final_stats.base_algorithm},
                              {"delta_levels", std::move(levels)},
                              {"compaction",
                               {{"duration_ms", compaction_ms},
                                {"peak_process_rss_bytes", peak_rss},
                                {"queries_served", searches_during_compaction},
                                {"query_failures", search_failures},
                                {"final_dense_memory_bytes", final_stats.resident_bytes}}}};
    } catch (const std::exception& error) {
        return rag::fail<nlohmann::json>(rag::Errc::invalid_argument,
                                         "runtime evaluation failed: " + std::string(error.what()));
    }
}

} // namespace lrs::eval
