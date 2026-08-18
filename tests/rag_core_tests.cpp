#include <rag/backend/candidate_backend.hpp>
#include <rag/backend/embedded_backend.hpp>
#include <rag/backend/embedded_checkpoint.hpp>
#if LRS_ENABLE_POSTGRES
#include <postgres/connection_pool.hpp>
#include <rag/backend/postgres_backend.hpp>
#include <rag/backend/postgres_config.hpp>
#include <rag/ingestion/postgres_job_store.hpp>
#include <rag/ingestion/postgres_runtime.hpp>
#endif
#include <rag/c/rag.h>
#include <rag/core/keys.hpp>
#include <rag/dense/backends.hpp>
#include <rag/dense/exact_sidecar.hpp>
#include <rag/dense/faiss_index.hpp>
#include <rag/dense/faiss_sidecar.hpp>
#include <rag/dense/hnsw_sidecar.hpp>
#include <rag/dense/index.hpp>
#include <rag/dense/native_hnsw.hpp>
#include <rag/dense/policy.hpp>
#include <rag/dense/simd.hpp>
#include <rag/dense/tiered_index.hpp>
#include <rag/engine.hpp>
#include <rag/fusion/fuse.hpp>
#include <rag/ingestion/coordinator.hpp>
#include <rag/ingestion/embedded_durability.hpp>
#include <rag/ingestion/embedded_runtime.hpp>
#include <rag/ingestion/job_store.hpp>
#include <rag/lexical/bm25.hpp>
#include <rag/migration/contract.hpp>
#include <rag/migration/embedded_endpoint.hpp>
#if LRS_ENABLE_POSTGRES
#include <rag/migration/postgres_endpoint.hpp>
#endif
#include <rag/preparation/document_preparer.hpp>
#include <rag/retrieval/runtime.hpp>
#include <rag/store/container.hpp>
#include <rag/store/container_view.hpp>
#include <rag/store/format.hpp>
#include <rag/store/wal.hpp>
#include <rag/text/chunker.hpp>

#include <nlohmann/json.hpp>

#include "backend_contract_suite.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {
int failures = 0;
#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::cerr << __FILE__ << ':' << __LINE__ << ": check failed: " #condition << '\n';     \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

struct TemporaryDirectory {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("rag-owned-tests-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { std::filesystem::create_directories(path); }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

struct MockTransport final : rag::dense::HttpTransport {
    rag::Result<rag::dense::HttpResponse> response = rag::dense::HttpResponse{};
    mutable rag::dense::HttpRequest request;
    rag::Result<rag::dense::HttpResponse>
    post(const rag::dense::HttpRequest& value) const override {
        request = value;
        return response;
    }
};

struct FailingMigrationEndpoint final : rag::migration::Endpoint {
    std::unique_ptr<rag::migration::Endpoint> inner;
    std::size_t writes_before_failure = 0;
    std::size_t writes = 0;

    std::string description() const override { return inner->description(); }
    rag::Result<rag::migration::DocumentBatch> read_batch(std::string_view after,
                                                          std::size_t limit) const override {
        return inner->read_batch(after, limit);
    }
    rag::Result<std::optional<rag::migration::MigrationProgress>>
    progress(std::string_view run_id) const override {
        return inner->progress(run_id);
    }
    rag::Result<void>
    write_batch(std::string_view run_id, const rag::migration::CorpusAudit& source,
                const rag::migration::DocumentBatch& batch,
                const rag::migration::MigrationProgress& progress_value) override {
        if (writes++ >= writes_before_failure)
            return rag::fail<void>(rag::Errc::io_error, "injected migration interruption");
        return inner->write_batch(run_id, source, batch, progress_value);
    }
    rag::Result<void> finish(std::string_view run_id, const rag::migration::CorpusAudit& source,
                             const rag::migration::MigrationProgress& progress_value) override {
        return inner->finish(run_id, source, progress_value);
    }
    rag::Result<rag::backend::CandidateList> exact_candidates(rag::VectorView query,
                                                              std::size_t k) const override {
        return inner->exact_candidates(query, k);
    }
};

struct WrongCountEmbedder {
    std::size_t dimension() const { return 2; }
    std::string_view identity() const { return "wrong-count"; }
    rag::Result<std::vector<rag::Vector>> embed(std::span<const std::string>) const {
        return std::vector<rag::Vector>{};
    }
};

struct GateState {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false;
    bool released = false;
    std::atomic<std::size_t> calls{0};
};

struct GateEmbedder {
    std::shared_ptr<GateState> state;
    std::size_t dimension() const { return 8; }
    std::string_view identity() const { return "gate-hash-v1"; }
    std::size_t max_concurrency() const { return 1; }
    rag::Result<std::vector<rag::Vector>> embed(std::span<const std::string> texts) const {
        ++state->calls;
        for (const auto& text : texts) {
            if (text.find("FAIL") != std::string::npos)
                return rag::fail<std::vector<rag::Vector>>(rag::Errc::transport_error,
                                                           "controlled embedding failure");
            if (text.find("SLOW") != std::string::npos) {
                std::unique_lock lock(state->mutex);
                state->entered = true;
                state->changed.notify_all();
                state->changed.wait(lock, [&] { return state->released; });
            }
        }
        return rag::dense::HashEmbedder{dimension()}.embed(texts);
    }
};

struct HybridContractBackend final : rag::backend::CandidateBackend {
    mutable std::atomic<std::size_t> lexical_calls{0};
    mutable std::atomic<std::size_t> dense_calls{0};

    rag::Result<void> activate(rag::backend::PreparedDocument, std::uint64_t) override {
        return {};
    }
    rag::Result<bool> erase(std::string, std::uint64_t) override { return false; }
    rag::Result<rag::backend::CandidateList>
    lexical_candidates(const rag::backend::LexicalRequest&) const override {
        ++lexical_calls;
        return rag::backend::CandidateList{{"chk_lexical", 1.0F, rag::backend::ScoreType::bm25}};
    }
    rag::Result<rag::backend::CandidateList>
    dense_candidates(const rag::backend::DenseRequest&) const override {
        ++dense_calls;
        return rag::backend::CandidateList{{"chk_dense", 1.0F, rag::backend::ScoreType::cosine}};
    }
    rag::Result<std::vector<rag::backend::StoredChunk>>
    fetch(std::span<const std::string>, rag::backend::FetchOptions) const override {
        return std::vector<rag::backend::StoredChunk>{};
    }
    rag::Result<rag::backend::BackendStats> stats() const override {
        return rag::backend::BackendStats{};
    }
};

void rewrite_crc(std::string& blob) {
    const auto checksum = rag::store::crc32(std::string_view(blob).substr(0, blob.size() - 4));
    std::memcpy(blob.data() + blob.size() - 4, &checksum, sizeof(checksum));
}

void test_chunking() {
    rag::text::ChunkOptions options;
    options.max_lines = 20;
    options.overlap_lines = 1;
    options.heading_context = true;
    options.policy.target_tokens = 10;
    options.policy.max_tokens = 12;
    options.policy.overlap_tokens = 2;
    options.measure_tokens = [](std::string_view text) { return text.size(); };
    const std::string source = "# H\nalpha beta gamma\n\xF0\x9F\x8C\xB2 delta epsilon\nlast";
    const auto first = rag::text::chunk_document(rag::DocId{3}, source, options);
    const auto second = rag::text::chunk_document(rag::DocId{3}, source, options);
    CHECK(!first.empty());
    CHECK(first.size() == second.size());
    for (std::size_t index = 0; index < first.size(); ++index) {
        CHECK(first[index].text == second[index].text);
        CHECK(first[index].start_line <= first[index].end_line);
        CHECK(options.measure_tokens(first[index].text) <= options.policy.max_tokens);
    }
    CHECK(first.front().start_line == 0);
    CHECK(first.back().end_line == 3);
    CHECK(!rag::text::chunking_fingerprint(options).empty());
}

void test_backend_contract_types_and_exact_dense_index() {
    static_assert(std::is_same_v<decltype(rag::dense::VectorRecord{}.vector), rag::VectorView>);
    const rag::backend::MetadataFilter filter{{{"tenant", "north"}, {"visibility", "team"}}};
    CHECK(filter.matches({{"tenant", "north"}, {"visibility", "team"}, {"kind", "guide"}}));
    CHECK(!filter.matches({{"tenant", "north"}, {"visibility", "private"}}));
    CHECK(!filter.matches({{"tenant", "north"}}));

    const rag::backend::MetadataFilter any_tenant(rag::backend::MetadataFilter::Requirements{
        {"tenant", {"south", "north", "north"}}, {"visibility", {"team"}}});
    CHECK(any_tenant.required.at("tenant") == std::vector<std::string>({"north", "south"}));
    CHECK(any_tenant.matches({{"tenant", "south"}, {"visibility", "team"}}));
    CHECK(!any_tenant.matches({{"tenant", "south"}, {"visibility", "private"}}));
    CHECK(!any_tenant.matches({{"visibility", "team"}}));
    const rag::backend::MetadataFilter empty_allowed(
        rag::backend::MetadataFilter::Requirements{{"tenant", {}}});
    CHECK(!empty_allowed.matches({{"tenant", "north"}}));
    const rag::backend::MetadataFilter explicit_empty(rag::backend::MetadataFilter::Requirements{});
    CHECK(!explicit_empty.matches({{"tenant", "north"}}));
    CHECK(rag::backend::MetadataFilter{}.matches({{"tenant", "north"}}));

    HybridContractBackend candidate_backend;
    const rag::Vector hybrid_query{1.0F, 0.0F};
    const auto batch =
        candidate_backend.hybrid_candidates({{"alpha", 2, filter}, {hybrid_query, 2, filter}});
    CHECK(batch && batch->lexical.size() == 1 && batch->dense.size() == 1);
    CHECK(candidate_backend.lexical_calls == 1);
    CHECK(candidate_backend.dense_calls == 1);

    rag::dense::NativeExactIndex index;
    std::array<rag::Vector, 3> vectors{rag::Vector{0.0F, 1.0F}, rag::Vector{1.0F, 0.0F},
                                       rag::Vector{0.6F, 0.8F}};
    const std::array<rag::dense::VectorRecord, 3> rows{
        rag::dense::VectorRecord{"chk_b", vectors[0]},
        rag::dense::VectorRecord{"chk_a", vectors[1]},
        rag::dense::VectorRecord{"chk_c", vectors[2]}};
    CHECK(index.build(rows));
    // VectorSource is borrowed only for build(): later owner mutation cannot
    // change the index generation that was atomically published.
    vectors[1][0] = -1.0F;

    const rag::Vector query{1.0F, 0.0F};
    const auto all = index.search(query, {}, 2);
    CHECK(all && all->size() == 2);
    CHECK(all && (*all)[0].chunk == "chk_a");
    CHECK(all && (*all)[1].chunk == "chk_c");

    const std::vector<rag::backend::ChunkKey> allowed{"chk_b", "chk_c"};
    const auto selected = index.search(query, {allowed}, 5);
    CHECK(selected && selected->size() == 2);
    CHECK(selected && (*selected)[0].chunk == "chk_c");
    CHECK(selected && (*selected)[1].chunk == "chk_b");

    const auto stats = index.stats();
    CHECK(stats.exact);
    CHECK(stats.vectors == 3);
    CHECK(stats.dimension == 2);
    CHECK(stats.implementation == "native");

    const std::array<rag::Vector, 2> ragged_vectors{rag::Vector{1.0F, 0.0F}, rag::Vector{1.0F}};
    const std::array<rag::dense::VectorRecord, 2> ragged{
        rag::dense::VectorRecord{"chk_a", ragged_vectors[0]},
        rag::dense::VectorRecord{"chk_b", ragged_vectors[1]}};
    CHECK(!index.build(ragged));
    // A failed build does not destroy the last valid generation.
    CHECK(index.stats().vectors == 3);
    CHECK(!index.search(rag::Vector{1.0F}, {}, 1));
}

void test_tiered_dense_index_contract() {
    auto empty = rag::dense::TieredDenseIndex::empty();
    CHECK(empty);
    if (!empty)
        return;
    const std::array<rag::Vector, 2> initial_vectors{rag::Vector{1.0F, 0.0F},
                                                     rag::Vector{0.0F, 1.0F}};
    const std::array<rag::dense::VectorRecord, 2> initial{
        rag::dense::VectorRecord{"chk_a", initial_vectors[0]},
        rag::dense::VectorRecord{"chk_b", initial_vectors[1]}};
    auto delta = (*empty)->with_delta(initial, {}, 2);
    CHECK(delta && (*delta)->stats().base_vectors == 0);
    CHECK(delta && (*delta)->stats().delta_vectors == 2);
    if (!delta)
        return;
    auto base = (*delta)->compact(initial, 2);
    CHECK(base && (*base)->stats().base_vectors == 2);
    CHECK(base && (*base)->stats().delta_vectors == 0);
    if (!base)
        return;
    const rag::Vector update_vector{0.8F, 0.6F};
    const std::array<rag::dense::VectorRecord, 1> update{
        rag::dense::VectorRecord{"chk_c", update_vector}};
    auto mixed = (*base)->with_delta(update, {"chk_a"}, 2);
    CHECK(mixed && !(*mixed)->contains_base("chk_a"));
    CHECK(mixed && (*mixed)->contains_base("chk_b"));
    CHECK(mixed && (*mixed)->contains_delta("chk_c"));
    if (!mixed)
        return;
    const std::vector<rag::backend::ChunkKey> base_allowed{"chk_b"};
    const std::vector<rag::backend::ChunkKey> delta_allowed{"chk_c"};
    const auto results = (*mixed)->search(rag::Vector{1.0F, 0.0F}, base_allowed, delta_allowed, 2);
    CHECK(results && results->size() == 2);
    CHECK(results && (*results)[0].chunk == "chk_c");
    CHECK(results && (*results)[1].chunk == "chk_b");
}

void test_native_hnsw_dense_contract() {
    std::vector<rag::Vector> vectors;
    std::vector<std::string> keys;
    std::vector<rag::dense::VectorRecord> rows;
    constexpr std::size_t count = 128;
    vectors.reserve(count);
    keys.reserve(count);
    rows.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float angle = static_cast<float>(index) * 0.049087385F;
        vectors.push_back({std::cos(angle), std::sin(angle)});
        keys.push_back("chk_hnsw_" + std::to_string(index));
    }
    for (std::size_t index = 0; index < count; ++index)
        rows.push_back({keys[index], vectors[index]});

    rag::dense::NativeHnswIndex hnsw;
    CHECK(hnsw.build(rows));
    const auto stats = hnsw.stats();
    CHECK(stats.vectors == count);
    CHECK(stats.algorithm == "hnsw");
    CHECK(stats.primary_vector_bytes >= count * 2 * sizeof(float));
    CHECK(stats.compressed_vector_bytes == 0);

    const auto nearest = hnsw.search(vectors[17], {}, 5);
    CHECK(nearest && nearest->size() == 5);
    CHECK(nearest && nearest->front().chunk == keys[17]);

    const std::vector<rag::backend::ChunkKey> all_allowed(keys.begin(), keys.end());
    const auto all_filtered = hnsw.search(vectors[17], {all_allowed}, 5);
    CHECK(all_filtered && nearest && *all_filtered == *nearest);

    const std::vector<rag::backend::ChunkKey> selective{keys[73], keys[91], keys[117]};
    const auto filtered = hnsw.search(vectors[73], {selective}, 10);
    CHECK(filtered && filtered->size() == selective.size());
    CHECK(filtered && filtered->front().chunk == keys[73]);
    if (filtered)
        for (const auto& candidate : *filtered)
            CHECK(std::find(selective.begin(), selective.end(), candidate.chunk) !=
                  selective.end());

    rag::dense::NativeHnswIndex invalid({.neighbors = 1});
    CHECK(!invalid.build(rows));

    rag::dense::DensePolicy automatic;
    automatic.exact_threshold = count;
    const auto selected = rag::dense::build_dense_index(automatic, rows);
    CHECK(selected && (*selected)->stats().algorithm == "hnsw");
    automatic.exact_threshold = count + 1;
    const auto exact = rag::dense::build_dense_index(automatic, rows);
    CHECK(exact && (*exact)->stats().algorithm == "exact");
    CHECK(!rag::dense::parse_dense_implementation("other"));
    CHECK(!rag::dense::parse_dense_algorithm("factory-string"));
    rag::dense::DensePolicy unavailable_faiss;
    unavailable_faiss.implementation = rag::dense::DenseImplementation::faiss;
    unavailable_faiss.algorithm = rag::dense::DenseAlgorithm::flat;
#if !LRS_ENABLE_FAISS
    CHECK(!rag::dense::build_dense_index(unavailable_faiss, rows));
#endif
}

#if LRS_ENABLE_FAISS
void test_faiss_dense_contract() {
    std::vector<rag::Vector> vectors;
    std::vector<std::string> keys;
    std::vector<rag::dense::VectorRecord> rows;
    constexpr std::size_t count = 512;
    constexpr std::size_t dimension = 8;
    vectors.reserve(count);
    keys.reserve(count);
    rows.reserve(count);
    for (std::size_t row = 0; row < count; ++row) {
        rag::Vector vector(dimension);
        for (std::size_t component = 0; component < dimension; ++component)
            vector[component] = std::sin(static_cast<float>((row + 1) * (component + 3)));
        rag::dense::normalize(vector);
        vectors.push_back(std::move(vector));
        keys.push_back("chk_faiss_" + std::to_string(row));
    }
    for (std::size_t row = 0; row < count; ++row)
        rows.push_back({keys[row], vectors[row]});

    TemporaryDirectory temporary;
    rag::dense::DensePolicy flat_policy;
    flat_policy.implementation = rag::dense::DenseImplementation::faiss;
    flat_policy.algorithm = rag::dense::DenseAlgorithm::flat;
    const auto flat_built = rag::dense::build_dense_index(flat_policy, rows);
    CHECK(flat_built);
    const auto flat =
        flat_built ? std::dynamic_pointer_cast<rag::dense::FaissIndex>(*flat_built) : nullptr;
    CHECK(flat);
    const auto sidecar = rag::dense::faiss_sidecar_path((temporary.path / "faiss.ragdb").string(),
                                                        rag::dense::DenseAlgorithm::flat);
    const auto fingerprint = rag::dense::faiss_sidecar_fingerprint(
        rows, "test-embedding-v1", rag::dense::DenseAlgorithm::flat, flat_policy.faiss);
    CHECK(flat && rag::dense::write_faiss_sidecar(sidecar, fingerprint, *flat));
    const auto restored = rag::dense::load_faiss_sidecar(
        sidecar, fingerprint, keys, rag::dense::DenseAlgorithm::flat, flat_policy.faiss);
    CHECK(restored);
    const auto restored_nearest =
        restored ? (*restored)->search(vectors[7], {}, 10)
                 : rag::fail<rag::backend::CandidateList>(rag::Errc::unavailable, "not restored");
    CHECK(restored_nearest && restored_nearest->size() == 10);
    CHECK(restored_nearest && restored_nearest->front().chunk == keys[7]);
    CHECK(!rag::dense::load_faiss_sidecar(sidecar, "wrong-fingerprint", keys,
                                          rag::dense::DenseAlgorithm::flat, flat_policy.faiss));

    const std::vector<rag::backend::ChunkKey> selective{keys[7], keys[311], keys[509]};
    for (const auto algorithm : {rag::dense::DenseAlgorithm::flat, rag::dense::DenseAlgorithm::hnsw,
                                 rag::dense::DenseAlgorithm::ivf_sq8}) {
        rag::dense::DensePolicy policy;
        policy.implementation = rag::dense::DenseImplementation::faiss;
        policy.algorithm = algorithm;
        policy.faiss.ivf_lists = 4;
        policy.faiss.ivf_probes = 2;
        policy.faiss.minimum_training_vectors_per_list = 64;
        policy.faiss.pq_subquantizers = 2;
        auto index = rag::dense::build_dense_index(policy, rows);
        CHECK(index);
        if (!index)
            continue;
        CHECK((*index)->stats().implementation == "faiss");
        const auto nearest = (*index)->search(vectors[7], {}, 10);
        CHECK(nearest && nearest->size() == 10);
        const auto filtered = (*index)->search(vectors[7], {selective}, 10);
        CHECK(filtered && filtered->size() == selective.size());
        if (filtered)
            for (const auto& candidate : *filtered)
                CHECK(std::find(selective.begin(), selective.end(), candidate.chunk) !=
                      selective.end());
    }

    constexpr std::size_t pq_count = 256 * 39;
    std::vector<rag::Vector> pq_vectors;
    std::vector<std::string> pq_keys;
    std::vector<rag::dense::VectorRecord> pq_rows;
    pq_vectors.reserve(pq_count);
    pq_keys.reserve(pq_count);
    pq_rows.reserve(pq_count);
    for (std::size_t row = 0; row < pq_count; ++row) {
        rag::Vector vector(dimension);
        for (std::size_t component = 0; component < dimension; ++component)
            vector[component] = std::sin(static_cast<float>((row + 5) * (component + 7)) * 0.013F);
        rag::dense::normalize(vector);
        pq_vectors.push_back(std::move(vector));
        pq_keys.push_back("chk_faiss_pq_" + std::to_string(row));
    }
    for (std::size_t row = 0; row < pq_count; ++row)
        pq_rows.push_back({pq_keys[row], pq_vectors[row]});
    rag::dense::DensePolicy pq_policy;
    pq_policy.implementation = rag::dense::DenseImplementation::faiss;
    pq_policy.algorithm = rag::dense::DenseAlgorithm::ivf_pq;
    pq_policy.faiss.ivf_lists = 4;
    pq_policy.faiss.ivf_probes = 2;
    pq_policy.faiss.pq_subquantizers = 2;
    const auto pq = rag::dense::build_dense_index(pq_policy, pq_rows);
    CHECK(pq);
    if (pq) {
        const std::vector<rag::backend::ChunkKey> pq_allowed{pq_keys[9], pq_keys[7001]};
        const auto pq_filtered = (*pq)->search(pq_vectors[9], {pq_allowed}, 5);
        CHECK(pq_filtered && pq_filtered->size() == pq_allowed.size());
    }

    rag::dense::DensePolicy undersized;
    undersized.implementation = rag::dense::DenseImplementation::faiss;
    undersized.algorithm = rag::dense::DenseAlgorithm::ivf_sq8;
    CHECK(!rag::dense::build_dense_index(undersized, rows));
    rag::dense::DensePolicy indivisible;
    indivisible.implementation = rag::dense::DenseImplementation::faiss;
    indivisible.algorithm = rag::dense::DenseAlgorithm::ivf_pq;
    indivisible.faiss.ivf_lists = 4;
    indivisible.faiss.minimum_training_vectors_per_list = 64;
    indivisible.faiss.pq_subquantizers = 3;
    CHECK(!rag::dense::build_dense_index(indivisible, rows));
}
#endif

void test_reusable_embedded_backend_contracts() {
    const auto exercise = [](rag::backend::EmbeddedMaintenancePolicy policy, std::string name,
                             std::size_t filler_chunks = 0) {
        policy.automatic_compaction = false;
        rag::backend::EmbeddedBackend backend(policy);
        lrs::tests::CandidateBackendContractOptions contract;
        contract.key_prefix = "contract/" + name;
        contract.dimension = 16;
        contract.filler_chunks = filler_chunks;
        contract.publish_base = [&backend] { return backend.compact(); };
        const auto result = lrs::tests::run_candidate_backend_contract(backend, contract);
        CHECK(result);
        if (!result)
            std::cerr << name << ": " << result.error().message << '\n';
    };

    rag::backend::EmbeddedMaintenancePolicy exact;
    exact.dense.algorithm = rag::dense::DenseAlgorithm::exact;
    exercise(exact, "native-exact");

    rag::backend::EmbeddedMaintenancePolicy hnsw;
    hnsw.dense.algorithm = rag::dense::DenseAlgorithm::hnsw;
    exercise(hnsw, "native-hnsw");

#if LRS_ENABLE_FAISS
    for (const auto algorithm :
         {rag::dense::DenseAlgorithm::flat, rag::dense::DenseAlgorithm::hnsw,
          rag::dense::DenseAlgorithm::ivf_sq8, rag::dense::DenseAlgorithm::ivf_pq}) {
        rag::backend::EmbeddedMaintenancePolicy faiss;
        faiss.dense.implementation = rag::dense::DenseImplementation::faiss;
        faiss.dense.algorithm = algorithm;
        faiss.dense.faiss.pq_subquantizers = 4;
        const std::size_t filler = algorithm == rag::dense::DenseAlgorithm::ivf_pq    ? 9'981
                                   : algorithm == rag::dense::DenseAlgorithm::ivf_sq8 ? 9'981
                                                                                      : 0;
        exercise(faiss, "faiss-" + std::string(rag::dense::name(algorithm)), filler);
    }
#endif
}

void test_embedded_backend_contract() {
    rag::text::ChunkOptions chunking;
    chunking.max_lines = 1;
    chunking.overlap_lines = 0;
    rag::preparation::PrepareOptions options;
    options.chunking = chunking;
    options.embedding_batch_size = 2;
    const rag::dense::AnyEmbedder embedder{rag::dense::HashEmbedder{16}};

    auto north =
        rag::preparation::prepare_document("docs/north", "alpha orchard\nshared guide",
                                           {{"tenant", "north"}}, "North", options, &embedder);
    auto south = rag::preparation::prepare_document(
        "docs/south", "alpha harbor", {{"tenant", "south"}}, "South", options, &embedder);
    CHECK(north && north->chunks.size() == 2);
    CHECK(south && south->chunks.size() == 1);
    CHECK(north && north->chunks[0].key.rfind("chk_", 0) == 0);
    CHECK(north && north->content_hash.rfind("fnv1a64:", 0) == 0);

    rag::backend::EmbeddedMaintenancePolicy manual_policy;
    manual_policy.automatic_compaction = false;
    rag::backend::EmbeddedBackend backend(manual_policy);
    CHECK(north && backend.activate(std::move(*north), 1));
    CHECK(south && backend.activate(std::move(*south), 1));
    const auto initial_stats = backend.stats();
    CHECK(initial_stats && initial_stats->live_documents == 2);
    CHECK(initial_stats && initial_stats->live_chunks == 3);
    CHECK(initial_stats && initial_stats->dense_base_chunks == 0);
    CHECK(initial_stats && initial_stats->dense_delta_chunks == 3);
    CHECK(initial_stats && initial_stats->catalog_embedding_bytes == 0);
    CHECK(initial_stats && initial_stats->capabilities.atomic_document_activation);

    rag::backend::MetadataFilter north_only{{{"tenant", "north"}}};
    const auto lexical = backend.lexical_candidates({"alpha", 10, north_only});
    CHECK(lexical && lexical->size() == 1);
    if (lexical) {
        const std::vector<rag::backend::ChunkKey> keys{lexical->front().chunk};
        const auto fetched = backend.fetch(keys, {true, false});
        CHECK(fetched && fetched->size() == 1);
        CHECK(fetched && fetched->front().document == "docs/north");
        CHECK(fetched && fetched->front().embedding.empty());
    }

    auto query = embedder.embed_one("alpha");
    CHECK(query);
    if (query) {
        rag::dense::normalize(*query);
        const auto dense = backend.dense_candidates({*query, 10, north_only});
        CHECK(dense && dense->size() == 2);
        if (dense && !dense->empty()) {
            const std::vector<rag::backend::ChunkKey> one{dense->front().chunk};
            const auto without_vector = backend.fetch(one, {true, false});
            const auto with_vector = backend.fetch(one, {true, true});
            CHECK(without_vector && without_vector->size() == 1);
            CHECK(without_vector && without_vector->front().embedding.empty());
            CHECK(with_vector && with_vector->size() == 1);
            CHECK(with_vector && with_vector->front().embedding.size() == 16);
        }
        rag::backend::SearchRequest request;
        request.query = "alpha";
        request.embedding = *query;
        request.top_k = 2;
        request.candidate_pool = 8;
        request.filter = north_only;
        const auto results = rag::retrieval::search(backend, request);
        CHECK(results && !results->empty());
        if (results)
            for (const auto& result : *results) {
                CHECK(result.document_key == "docs/north");
                CHECK(result.metadata.at("tenant") == "north");
                CHECK(result.chunk_key.rfind("chk_", 0) == 0);
                CHECK(result.revision == 1);
            }
    }

    std::atomic<bool> stop_reader{false};
    std::atomic<bool> reader_failed{false};
    std::atomic<std::size_t> read_count{0};
    std::thread reader;
    if (query) {
        reader = std::thread([&] {
            while (!stop_reader.load()) {
                const auto found = backend.dense_candidates({*query, 3, {}});
                if (!found || found->empty())
                    reader_failed = true;
                ++read_count;
            }
        });
        while (read_count.load() == 0)
            std::this_thread::yield();
    }
    CHECK(backend.compact());
    stop_reader = true;
    if (reader.joinable())
        reader.join();
    CHECK(!reader_failed.load());
    const auto compacted = backend.stats();
    CHECK(compacted && compacted->dense_base_chunks == 3);
    CHECK(compacted && compacted->dense_delta_chunks == 0);
    CHECK(compacted && compacted->tombstones == 0);
    CHECK(compacted && compacted->dense_algorithm == "exact");

    auto east = rag::preparation::prepare_document(
        "docs/east", "alpha estuary", {{"tenant", "north"}}, "East", options, &embedder);
    CHECK(east && backend.activate(std::move(*east), 1));
    if (query) {
        const auto merged = backend.dense_candidates({*query, 10, north_only});
        CHECK(merged && merged->size() == 3);
        if (merged) {
            std::vector<rag::backend::ChunkKey> keys;
            for (const auto& candidate : *merged)
                keys.push_back(candidate.chunk);
            const auto resolved = backend.fetch(keys, {true, false});
            CHECK(resolved && resolved->size() == 3);
            if (resolved)
                for (const auto& chunk : *resolved)
                    CHECK(chunk.metadata.at("tenant") == "north");
        }
    }

    auto replacement = rag::preparation::prepare_document(
        "docs/north", "replacement cedar", {{"tenant", "north"}}, "North v2", options, &embedder);
    CHECK(replacement);
    const auto old_key = lexical && !lexical->empty() ? lexical->front().chunk : "";
    CHECK(replacement && backend.activate(std::move(*replacement), 2));
    const auto with_tombstones = backend.stats();
    CHECK(with_tombstones && with_tombstones->dense_base_chunks == 3);
    CHECK(with_tombstones && with_tombstones->dense_delta_chunks == 2);
    CHECK(with_tombstones && with_tombstones->tombstones == 2);
    CHECK(with_tombstones && with_tombstones->maintenance_required);
    const auto old_term = backend.lexical_candidates({"orchard", 10, {}});
    CHECK(old_term && old_term->empty());
    if (!old_key.empty()) {
        const std::vector<rag::backend::ChunkKey> keys{old_key};
        const auto old = backend.fetch(keys, {});
        CHECK(old && old->empty());
    }

    auto stale =
        rag::preparation::prepare_document("docs/north", "stale", {}, "", options, &embedder);
    CHECK(stale && !backend.activate(std::move(*stale), 1));
    const auto still_replacement = backend.lexical_candidates({"replacement", 10, {}});
    CHECK(still_replacement && still_replacement->size() == 1);

    auto malformed = rag::preparation::prepare_document("docs/north", "bad\npartial", {}, "",
                                                        options, &embedder);
    CHECK(malformed && malformed->chunks.size() == 2);
    if (malformed) {
        malformed->chunks.back().embedding.clear();
        CHECK(!backend.activate(std::move(*malformed), 3));
    }
    const auto after_malformed = backend.lexical_candidates({"replacement", 10, {}});
    CHECK(after_malformed && after_malformed->size() == 1);

    CHECK(backend.compact());
    const auto recompacted = backend.stats();
    CHECK(recompacted && recompacted->dense_base_chunks == 3);
    CHECK(recompacted && recompacted->dense_delta_chunks == 0);
    CHECK(recompacted && recompacted->tombstones == 0);

    const auto erased = backend.erase("docs/north", 3);
    CHECK(erased && *erased);
    const auto after_delete_stats = backend.stats();
    CHECK(after_delete_stats && after_delete_stats->live_documents == 2);
    CHECK(after_delete_stats && after_delete_stats->dense_delta_chunks == 0);
    CHECK(after_delete_stats && after_delete_stats->tombstones == 1);
    const auto after_erase = backend.lexical_candidates({"replacement", 10, {}});
    CHECK(after_erase && after_erase->empty());
    auto superseded =
        rag::preparation::prepare_document("docs/north", "late", {}, "", options, &embedder);
    CHECK(superseded && !backend.activate(std::move(*superseded), 2));

    rag::backend::EmbeddedMaintenancePolicy constrained_policy;
    constrained_policy.automatic_compaction = false;
    constrained_policy.delta_chunk_limit = 10'000;
    constrained_policy.compaction_memory_budget = 1;
    rag::backend::EmbeddedBackend constrained(constrained_policy);
    auto budgeted = rag::preparation::prepare_document("docs/budget", "budget birch", {}, "",
                                                       options, &embedder);
    CHECK(budgeted && constrained.activate(std::move(*budgeted), 1));
    const auto constrained_stats = constrained.stats();
    CHECK(constrained_stats && constrained_stats->maintenance_required);
    CHECK(!constrained.compact());
    const auto before_override = constrained.stats();
    CHECK(before_override && before_override->dense_delta_chunks == 1);
    CHECK(constrained.compact(true));
    const auto after_override = constrained.stats();
    CHECK(after_override && after_override->dense_base_chunks == 1);
    CHECK(after_override && after_override->dense_delta_chunks == 0);

    rag::backend::EmbeddedMaintenancePolicy hnsw_policy;
    hnsw_policy.automatic_compaction = false;
    hnsw_policy.dense.algorithm = rag::dense::DenseAlgorithm::hnsw;
    rag::backend::EmbeddedBackend hnsw_backend(hnsw_policy);
    auto hnsw_document =
        rag::preparation::prepare_document("docs/hnsw", "hnsw hazel", {}, "", options, &embedder);
    CHECK(hnsw_document && hnsw_backend.activate(std::move(*hnsw_document), 1));
    CHECK(hnsw_backend.compact());
    const auto hnsw_stats = hnsw_backend.stats();
    CHECK(hnsw_stats && hnsw_stats->dense_implementation == "native");
    CHECK(hnsw_stats && hnsw_stats->dense_algorithm == "hnsw");
    CHECK(hnsw_stats && !hnsw_stats->dense_exact);
}

#if LRS_ENABLE_POSTGRES
void test_postgres_backend_contract() {
    rag::backend::PostgresConfig invalid;
    invalid.connection_string = "host=database.example dbname=rag sslmode=require";
    CHECK(!rag::backend::validate_postgres_config(invalid));
    invalid.connection_string =
        "host=database.example dbname=rag sslmode=verify-full sslrootcert=/tmp/ca.pem";
    CHECK(rag::backend::validate_postgres_config(invalid));
    invalid.schema = "not-valid";
    CHECK(!rag::backend::validate_postgres_config(invalid));

    const char* connection_string = std::getenv("LRS_TEST_POSTGRES_URL");
    if (!connection_string || *connection_string == '\0')
        return;
    rag::backend::PostgresConfig config;
    config.connection_string = connection_string;
    config.schema = "lrs_contract";
    config.corpus =
        "corpus_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    config.pool_size = 2;
    rag::backend::PostgresConfig pool_config = config;
    pool_config.pool_size = 1;
    pool_config.statement_timeout = std::chrono::milliseconds(50);
    auto pool = rag::postgres::ConnectionPool::open(pool_config, nullptr, nullptr);
    CHECK(pool);
    if (pool) {
        {
            auto held = (*pool)->acquire(std::chrono::milliseconds(50));
            CHECK(held);
            const auto saturated = (*pool)->acquire(std::chrono::milliseconds(20));
            CHECK(!saturated && saturated.error().code == rag::Errc::unavailable);
        }
        auto timed = (*pool)->acquire(std::chrono::milliseconds(50));
        CHECK(timed);
        if (timed) {
            const auto cancelled = timed->connection().execute("SELECT pg_sleep(0.2)");
            CHECK(!cancelled && cancelled.error().code == rag::Errc::unavailable);
        }
    }
    auto backend = rag::backend::PostgresBackend::open(config);
    CHECK(backend);
    if (!backend)
        return;
    auto jobs = rag::ingestion::PostgresJobStore::open(config);
    CHECK(jobs);
    if (!jobs)
        return;

    rag::backend::PostgresConfig shared_contract_config = config;
    shared_contract_config.corpus += "_shared_contract";
    auto shared_contract_backend = rag::backend::PostgresBackend::open(shared_contract_config);
    CHECK(shared_contract_backend);
    if (shared_contract_backend) {
        lrs::tests::CandidateBackendContractOptions shared_contract;
        shared_contract.key_prefix = "contract/postgres-exact";
        shared_contract.dimension = 16;
        shared_contract.require_durable = true;
        const auto verified =
            lrs::tests::run_candidate_backend_contract(**shared_contract_backend, shared_contract);
        CHECK(verified);
        if (!verified)
            std::cerr << "postgres-exact: " << verified.error().message << '\n';
    }

    rag::backend::PostgresConfig hnsw_contract_config = config;
    hnsw_contract_config.corpus += "_hnsw_contract";
    hnsw_contract_config.vector_index = rag::backend::PostgresVectorIndex::hnsw;
    auto hnsw_contract_backend = rag::backend::PostgresBackend::open(hnsw_contract_config);
    CHECK(hnsw_contract_backend);
    if (hnsw_contract_backend) {
        lrs::tests::CandidateBackendContractOptions hnsw_contract;
        hnsw_contract.key_prefix = "contract/postgres-hnsw";
        hnsw_contract.dimension = 16;
        hnsw_contract.require_durable = true;
        const auto verified =
            lrs::tests::run_candidate_backend_contract(**hnsw_contract_backend, hnsw_contract);
        CHECK(verified);
        if (!verified)
            std::cerr << "postgres-hnsw: " << verified.error().message << '\n';
    }

    rag::preparation::PrepareOptions options;
    options.chunking.max_lines = 1;
    options.chunking.overlap_lines = 0;
    const rag::dense::AnyEmbedder embedder{rag::dense::HashEmbedder{16}};
    auto north =
        rag::preparation::prepare_document("docs/postgres-north", "alpha orchard\nshared guide",
                                           {{"tenant", "north"}}, "North", options, &embedder);
    auto south = rag::preparation::prepare_document(
        "docs/postgres-south", "alpha harbor", {{"tenant", "south"}}, "South", options, &embedder);
    CHECK(north && (*backend)->activate(*north, 1));
    CHECK(south && (*backend)->activate(*south, 1));
    const auto stats = (*backend)->stats();
    CHECK(stats && stats->live_documents == 2);
    CHECK(stats && stats->live_chunks == 3);
    CHECK(stats && stats->embedding_dimension == 16);
    CHECK(stats && stats->capabilities.durable && stats->capabilities.atomic_document_activation);

    rag::backend::EmbeddedBackend lexical_oracle;
    CHECK(north && lexical_oracle.activate(*north, 1));
    CHECK(south && lexical_oracle.activate(*south, 1));
    const auto postgres_bm25 = (*backend)->lexical_candidates({"alpha alpha shared", 10, {}});
    const auto embedded_bm25 = lexical_oracle.lexical_candidates({"alpha alpha shared", 10, {}});
    CHECK(postgres_bm25 && embedded_bm25);
    CHECK(postgres_bm25 && embedded_bm25 && postgres_bm25->size() == embedded_bm25->size());
    if (postgres_bm25 && embedded_bm25 && postgres_bm25->size() == embedded_bm25->size()) {
        for (std::size_t index = 0; index < postgres_bm25->size(); ++index) {
            CHECK((*postgres_bm25)[index].chunk == (*embedded_bm25)[index].chunk);
            CHECK(std::abs((*postgres_bm25)[index].raw_score - (*embedded_bm25)[index].raw_score) <
                  1.0e-5f);
        }
    }

    auto fault_connection = rag::postgres::Connection::open(config);
    CHECK(fault_connection);
    if (fault_connection) {
        const std::string function =
            "CREATE OR REPLACE FUNCTION \"" + config.schema +
            "\".lrs_reject_rollback_posting() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
            "IF NEW.term = 'rollback' THEN RAISE EXCEPTION 'injected activation failure'; "
            "END IF; RETURN NEW; END $$";
        const std::string trigger =
            "CREATE TRIGGER lrs_reject_rollback_posting BEFORE INSERT ON \"" + config.schema +
            "\".postings FOR EACH ROW EXECUTE FUNCTION \"" + config.schema +
            "\".lrs_reject_rollback_posting()";
        CHECK(fault_connection->execute(function));
        CHECK(fault_connection->execute(trigger));
        auto rollback_document = rag::preparation::prepare_document(
            "docs/rollback", "rollback sentinel", {}, "", options, &embedder);
        CHECK(rollback_document && !(*backend)->activate(*rollback_document, 1));
        CHECK(fault_connection->execute("DROP TRIGGER lrs_reject_rollback_posting ON \"" +
                                        config.schema + "\".postings"));
        CHECK(fault_connection->execute("DROP FUNCTION \"" + config.schema +
                                        "\".lrs_reject_rollback_posting()"));
        const auto after_rollback = (*backend)->stats();
        CHECK(after_rollback && after_rollback->live_documents == 2);
        CHECK(after_rollback && after_rollback->live_chunks == 3);
        const auto invisible = (*backend)->lexical_candidates({"sentinel", 10, {}});
        CHECK(invisible && invisible->empty());
    }

    const rag::backend::MetadataFilter north_only{{{"tenant", "north"}}};
    const auto lexical = (*backend)->lexical_candidates({"alpha", 10, north_only});
    CHECK(lexical && lexical->size() == 1);
    auto query = embedder.embed_one("alpha");
    CHECK(query);
    if (!query)
        return;
    rag::dense::normalize(*query);
    const auto dense = (*backend)->dense_candidates({*query, 10, north_only});
    CHECK(dense && dense->size() == 2);
    const auto hybrid =
        (*backend)->hybrid_candidates({{"alpha", 10, north_only}, {*query, 10, north_only}});
    CHECK(hybrid && hybrid->lexical.size() == 1 && hybrid->dense.size() == 2);

    rag::backend::SearchRequest request;
    request.query = "alpha";
    request.embedding = *query;
    request.top_k = 2;
    request.candidate_pool = 8;
    request.filter = north_only;
    const auto results = rag::retrieval::search(**backend, request);
    CHECK(results && !results->empty());
    if (results)
        for (const auto& result : *results)
            CHECK(result.document_key == "docs/postgres-north");

    auto replacement = rag::preparation::prepare_document(
        "docs/postgres-north", "replacement cedar", {{"tenant", "north"}}, "Replacement", options,
        &embedder);
    CHECK(replacement && (*backend)->activate(*replacement, 2));
    CHECK(replacement && (*backend)->activate(*replacement, 2));
    CHECK(north && !(*backend)->activate(*north, 1));
    const auto old = (*backend)->lexical_candidates({"orchard", 10, {}});
    const auto current = (*backend)->lexical_candidates({"cedar", 10, {}});
    CHECK(old && old->empty());
    CHECK(current && current->size() == 1);

    rag::ingestion::IngestionJob durable_job;
    durable_job.id = "job_postgres_contract";
    durable_job.input.document = replacement ? replacement->key : "docs/postgres-north";
    durable_job.input.title = replacement ? replacement->title : "Replacement";
    durable_job.input.content = replacement ? replacement->content : "replacement cedar";
    durable_job.input.metadata = replacement ? replacement->metadata : rag::Metadata{};
    durable_job.revision = 2;
    durable_job.content_hash = replacement ? replacement->content_hash : "hash";
    durable_job.status = rag::ingestion::JobStatus::queued;
    durable_job.created_at_ms = durable_job.updated_at_ms = 1;
    CHECK((*jobs)->persist(durable_job));
    durable_job.status = rag::ingestion::JobStatus::ready;
    if (replacement)
        durable_job.prepared = *replacement;
    durable_job.updated_at_ms = 2;
    CHECK((*jobs)->persist(durable_job));
    const auto loaded_jobs = (*jobs)->load_latest();
    CHECK(loaded_jobs && loaded_jobs->size() == 1);
    CHECK(loaded_jobs && loaded_jobs->front().status == rag::ingestion::JobStatus::ready);
    CHECK(loaded_jobs && loaded_jobs->front().prepared.has_value());

    const rag::dense::AnyEmbedder incompatible_embedder{rag::dense::HashEmbedder{8}};
    auto incompatible = rag::preparation::prepare_document("docs/incompatible", "should rollback",
                                                           {}, "", options, &incompatible_embedder);
    CHECK(incompatible && !(*backend)->activate(*incompatible, 1));
    const auto after_failed = (*backend)->stats();
    CHECK(after_failed && after_failed->live_documents == 2);

    CHECK((*backend)->erase("docs/postgres-north", 3));
    const auto erased_again = (*backend)->erase("docs/postgres-north", 3);
    CHECK(erased_again && !*erased_again);
    CHECK(replacement && !(*backend)->activate(*replacement, 2));
    const auto deleted = (*backend)->lexical_candidates({"cedar", 10, {}});
    CHECK(deleted && deleted->empty());

    rag::ingestion::PostgresRuntimeConfig runtime_config;
    runtime_config.database = config;
    runtime_config.database.corpus += "_runtime";
    runtime_config.preparation = options;
    runtime_config.embedder = embedder;
    runtime_config.coordinator.worker_count = 1;
    runtime_config.coordinator.queue_capacity = 4;
    auto runtime = rag::ingestion::PostgresRuntime::open(runtime_config);
    CHECK(runtime);
    if (runtime) {
        const auto submitted = (*runtime)->submit(
            {"docs/runtime", "Runtime", "persistent spruce", {{"tenant", "runtime"}}}, false);
        CHECK(submitted && submitted->job.status == rag::ingestion::JobStatus::ready);
        const auto job_id = submitted ? submitted->job.id : std::string{};
        (*runtime).reset();
        auto reopened = rag::ingestion::PostgresRuntime::open(runtime_config);
        CHECK(reopened);
        CHECK(reopened && (*reopened)->job(job_id));
        if (reopened) {
            rag::backend::SearchRequest persisted;
            persisted.query = "spruce";
            persisted.mode = rag::backend::SearchMode::lexical;
            persisted.top_k = 2;
            const auto found = (*reopened)->search(std::move(persisted));
            CHECK(found && found->size() == 1);
            CHECK(found && found->front().document_key == "docs/runtime");
        }
    }

    rag::backend::PostgresConfig hnsw_config = config;
    hnsw_config.corpus += "_hnsw";
    hnsw_config.vector_index = rag::backend::PostgresVectorIndex::hnsw;
    hnsw_config.hnsw_ef_search = 80;
    auto hnsw = rag::backend::PostgresBackend::open(hnsw_config);
    CHECK(hnsw);
    CHECK(hnsw && north && (*hnsw)->activate(*north, 1));
    CHECK(hnsw && south && (*hnsw)->activate(*south, 1));
    const auto hnsw_dense =
        hnsw ? (*hnsw)->dense_candidates({*query, 10, north_only})
             : rag::fail<rag::backend::CandidateList>(rag::Errc::unavailable, "not opened");
    CHECK(hnsw_dense && hnsw_dense->size() == 2);
    const auto hnsw_stats =
        hnsw ? (*hnsw)->stats()
             : rag::fail<rag::backend::BackendStats>(rag::Errc::unavailable, "not opened");
    CHECK(hnsw_stats && !hnsw_stats->dense_exact && hnsw_stats->dense_algorithm == "hnsw");

    TemporaryDirectory migration_files;
    const auto embedded_source_path = migration_files.path / "source.ragdb";
    const auto embedded_roundtrip_path = migration_files.path / "roundtrip.ragdb";
    rag::backend::EmbeddedMaintenancePolicy migration_policy;
    migration_policy.automatic_compaction = false;
    rag::backend::EmbeddedBackend embedded_source_backend(migration_policy);
    CHECK(north && embedded_source_backend.activate(*north, 1));
    CHECK(south && embedded_source_backend.activate(*south, 1));
    CHECK(embedded_source_backend.checkpoint(embedded_source_path.string(), 0));
    std::ifstream source_input(embedded_source_path, std::ios::binary);
    const std::string source_bytes((std::istreambuf_iterator<char>(source_input)), {});
    source_input.close();
    auto embedded_source = rag::migration::open_embedded_source(embedded_source_path.string());
    rag::backend::PostgresConfig migration_config = config;
    migration_config.corpus += "_migration";
    auto postgres_destination = rag::migration::open_postgres_destination(migration_config);
    CHECK(embedded_source && postgres_destination);
    const auto to_postgres =
        embedded_source && postgres_destination
            ? rag::migration::migrate(**embedded_source, **postgres_destination,
                                      "embedded-to-postgres", {1, 2})
            : rag::fail<rag::migration::MigrationReport>(rag::Errc::unavailable,
                                                         "migration endpoints unavailable");
    CHECK(to_postgres && to_postgres->complete);
    CHECK(to_postgres && to_postgres->source_audit == to_postgres->destination_audit);
    std::ifstream source_after_input(embedded_source_path, std::ios::binary);
    const std::string source_after((std::istreambuf_iterator<char>(source_after_input)), {});
    CHECK(source_after == source_bytes);

    auto migrated_backend = rag::backend::PostgresBackend::open(migration_config);
    const auto before_reverse =
        migrated_backend ? (*migrated_backend)->stats()
                         : rag::fail<rag::backend::BackendStats>(rag::Errc::unavailable,
                                                                 "migration backend unavailable");
    auto postgres_source = rag::migration::open_postgres_source(migration_config);
    auto embedded_destination =
        rag::migration::open_embedded_destination(embedded_roundtrip_path.string());
    CHECK(postgres_source && embedded_destination);
    const auto to_embedded =
        postgres_source && embedded_destination
            ? rag::migration::migrate(**postgres_source, **embedded_destination,
                                      "postgres-to-embedded", {1, 2})
            : rag::fail<rag::migration::MigrationReport>(rag::Errc::unavailable,
                                                         "migration endpoints unavailable");
    CHECK(to_embedded && to_embedded->complete);
    CHECK(to_embedded && to_embedded->source_audit == to_embedded->destination_audit);
    const auto after_reverse =
        migrated_backend ? (*migrated_backend)->stats()
                         : rag::fail<rag::backend::BackendStats>(rag::Errc::unavailable,
                                                                 "migration backend unavailable");
    CHECK(before_reverse && after_reverse &&
          before_reverse->generation == after_reverse->generation);
    const auto roundtrip =
        rag::backend::EmbeddedCheckpointStore::load(embedded_roundtrip_path.string());
    CHECK(roundtrip && roundtrip->documents.size() == 2);
    CHECK(roundtrip && north &&
          roundtrip->documents.front().prepared->chunks.front().key == north->chunks.front().key);

    rag::backend::PostgresConfig unavailable = config;
    unavailable.connection_string = "host=127.0.0.1 port=1 dbname=missing connect_timeout=1";
    CHECK(!rag::backend::PostgresBackend::open(unavailable));
}
#endif

void test_automatic_embedded_compaction() {
    rag::backend::EmbeddedMaintenancePolicy policy;
    policy.delta_chunk_limit = 1;
    policy.automatic_compaction = true;
    rag::backend::EmbeddedBackend backend(policy);

    rag::preparation::PrepareOptions options;
    options.chunking.max_lines = 1;
    options.chunking.overlap_lines = 0;
    const rag::dense::AnyEmbedder embedder{rag::dense::HashEmbedder{16}};
    auto document = rag::preparation::prepare_document("docs/automatic", "automatic ash", {}, "",
                                                       options, &embedder);
    CHECK(document && backend.activate(std::move(*document), 1));

    bool compacted = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto stats = backend.stats();
        if (stats && stats->dense_base_chunks == 1 && stats->dense_delta_chunks == 0) {
            compacted = true;
            break;
        }
        std::this_thread::yield();
    }
    CHECK(compacted);
}

void test_embedded_checkpoint_contract() {
    TemporaryDirectory temporary;
    const auto checkpoint = temporary.path / "knowledge.ragdb";
    rag::backend::EmbeddedMaintenancePolicy policy;
    policy.automatic_compaction = false;
    rag::backend::EmbeddedBackend backend(policy);
    rag::preparation::PrepareOptions options;
    options.chunking.max_lines = 1;
    options.chunking.overlap_lines = 0;
    const rag::dense::AnyEmbedder embedder{rag::dense::HashEmbedder{16}};

    auto first = rag::preparation::prepare_document(
        "docs/checkpoint", "old oak", {{"tenant", "north"}}, "Old", options, &embedder);
    auto replacement = rag::preparation::prepare_document(
        "docs/checkpoint", "new maple", {{"tenant", "north"}}, "New", options, &embedder);
    CHECK(first && backend.activate(std::move(*first), 1));
    const auto replacement_copy = replacement;
    CHECK(replacement && backend.activate(std::move(*replacement), 2));
    auto deleted = rag::preparation::prepare_document("docs/deleted", "gone birch", {}, "", options,
                                                      &embedder);
    CHECK(deleted && backend.activate(std::move(*deleted), 1));
    CHECK(backend.erase("docs/deleted", 2));
    const auto before = backend.lexical_candidates({"maple", 10, {}});
    CHECK(before && before->size() == 1);
    const std::string stable_key = before && !before->empty() ? before->front().chunk : "";
    CHECK(!backend.checkpoint((temporary.path / "missing" / "failed.ragdb").string(), 0));
    const auto after_failed_checkpoint = backend.lexical_candidates({"maple", 10, {}});
    CHECK(after_failed_checkpoint && after_failed_checkpoint->size() == 1);
    const auto jobs_path = temporary.path / "knowledge.jobs";
    auto jobs =
        rag::ingestion::AppendOnlyJobStore::open(jobs_path.string(), rag::store::SyncMode::none);
    CHECK(jobs && replacement_copy);
    if (!jobs || !replacement_copy)
        return;
    rag::ingestion::IngestionJob ready;
    ready.id = "job_checkpoint_ready";
    ready.input.document = replacement_copy->key;
    ready.revision = 2;
    ready.content_hash = replacement_copy->content_hash;
    ready.status = rag::ingestion::JobStatus::ready;
    ready.created_at_ms = ready.updated_at_ms = 1;
    ready.prepared = *replacement_copy;
    CHECK((*jobs)->persist(ready));
    const auto represented_position = (*jobs)->size_bytes();
    CHECK(rag::ingestion::checkpoint_embedded(backend, **jobs, checkpoint.string()));
    const auto sidecar = rag::dense::exact_sidecar_path(checkpoint.string());
    CHECK(std::filesystem::exists(sidecar));
    CHECK(backend.checkpoint_wal_position() == represented_position);
    CHECK((*jobs)->size_bytes() > 0);

    const auto legacy = rag::index::Corpus::load(checkpoint.string());
    CHECK(legacy);
    auto reopened = rag::backend::EmbeddedBackend::open_checkpoint(checkpoint.string(), policy);
    CHECK(reopened);
    if (!reopened)
        return;
    CHECK((*reopened)->checkpoint_wal_position() == represented_position);
    const auto reopened_stats = (*reopened)->stats();
    CHECK(reopened_stats && reopened_stats->dense_mapped_bytes > 0);
    const auto after = (*reopened)->lexical_candidates({"maple", 10, {}});
    CHECK(after && after->size() == 1);
    CHECK(after && after->front().chunk == stable_key);
    const auto old = (*reopened)->lexical_candidates({"oak", 10, {}});
    CHECK(old && old->empty());
    CHECK(replacement_copy && (*reopened)->activate(*replacement_copy, 2));
    auto stale = rag::preparation::prepare_document("docs/checkpoint", "stale elm", {}, "", options,
                                                    &embedder);
    CHECK(stale && !(*reopened)->activate(std::move(*stale), 2));
    auto resurrect = rag::preparation::prepare_document("docs/deleted", "resurrected elm", {}, "",
                                                        options, &embedder);
    CHECK(resurrect && !(*reopened)->activate(std::move(*resurrect), 2));

    (*reopened).reset();
    std::ofstream corrupt_sidecar(sidecar, std::ios::binary | std::ios::trunc);
    corrupt_sidecar << "not a dense cache";
    corrupt_sidecar.close();
    auto rebuilt = rag::backend::EmbeddedBackend::open_checkpoint(checkpoint.string(), policy);
    CHECK(rebuilt);
    if (rebuilt) {
        const auto rebuilt_stats = (*rebuilt)->stats();
        CHECK(rebuilt_stats && rebuilt_stats->dense_mapped_bytes > 0);
        auto query = embedder.embed_one("maple");
        CHECK(query);
        if (query) {
            rag::dense::normalize(*query);
            const auto dense = (*rebuilt)->dense_candidates({*query, 10, {}});
            CHECK(dense && dense->size() == 1);
            CHECK(dense && dense->front().chunk == stable_key);
        }
    }

    std::fstream corrupt(checkpoint, std::ios::binary | std::ios::in | std::ios::out);
    CHECK(corrupt.good());
    corrupt.seekp(8);
    corrupt.put('\x7f');
    corrupt.close();
    CHECK(!rag::backend::EmbeddedBackend::open_checkpoint(checkpoint.string(), policy));
}

void test_embedded_migration_contract() {
    TemporaryDirectory temporary;
    const auto source_path = temporary.path / "source.ragdb";
    const auto destination_path = temporary.path / "destination.ragdb";
    rag::backend::EmbeddedMaintenancePolicy policy;
    policy.automatic_compaction = false;
    rag::backend::EmbeddedBackend backend(policy);
    rag::preparation::PrepareOptions options;
    options.chunking.max_lines = 1;
    options.chunking.overlap_lines = 0;
    const rag::dense::AnyEmbedder embedder{rag::dense::HashEmbedder{16}};
    auto checkpoint_document = rag::preparation::prepare_document(
        "docs/a", "alpha ash", {{"tenant", "north"}}, "Alpha", options, &embedder);
    auto ready_document = rag::preparation::prepare_document(
        "docs/b", "beta birch\nshared branch", {{"tenant", "south"}}, "Beta", options, &embedder);
    auto failed_document = rag::preparation::prepare_document("docs/c", "excluded cedar", {},
                                                              "Failed", options, &embedder);
    CHECK(checkpoint_document && backend.activate(*checkpoint_document, 1));
    CHECK(backend.checkpoint(source_path.string(), 0));
    auto jobs = rag::ingestion::AppendOnlyJobStore::open(source_path.string() + ".jobs",
                                                         rag::store::SyncMode::full);
    CHECK(jobs && ready_document && failed_document);
    if (!jobs || !ready_document || !failed_document)
        return;
    rag::ingestion::IngestionJob ready;
    ready.id = "job_migration_ready";
    ready.input.document = ready_document->key;
    ready.revision = 1;
    ready.content_hash = ready_document->content_hash;
    ready.status = rag::ingestion::JobStatus::ready;
    ready.created_at_ms = ready.updated_at_ms = 1;
    ready.prepared = *ready_document;
    CHECK((*jobs)->persist(ready));
    rag::ingestion::IngestionJob failed = ready;
    failed.id = "job_migration_failed";
    failed.input.document = failed_document->key;
    failed.content_hash = failed_document->content_hash;
    failed.status = rag::ingestion::JobStatus::failed;
    failed.prepared.reset();
    failed.updated_at_ms = 2;
    CHECK((*jobs)->persist(failed));
    (*jobs).reset();

    const auto bytes = [](const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    };
    const auto source_before = bytes(source_path);
    const auto jobs_before = bytes(source_path.string() + ".jobs");
    auto source = rag::migration::open_embedded_source(source_path.string());
    auto destination = rag::migration::open_embedded_destination(destination_path.string());
    CHECK(source && destination);
    if (!source || !destination)
        return;
    const auto source_audit = rag::migration::audit(**source, 1);
    CHECK(source_audit && source_audit->documents == 2 && source_audit->chunks == 3);

    const auto invalid_path = temporary.path / "invalid-id.ragdb";
    auto invalid_document = *checkpoint_document;
    invalid_document.chunks.front().key = "chk_not_stable";
    rag::backend::EmbeddedCheckpoint invalid_checkpoint;
    invalid_checkpoint.documents.push_back(
        {std::make_shared<rag::backend::PreparedDocument>(std::move(invalid_document)), 1});
    invalid_checkpoint.revisions.emplace("docs/a", 1);
    CHECK(rag::backend::EmbeddedCheckpointStore::save(invalid_path.string(), invalid_checkpoint));
    auto invalid_source = rag::migration::open_embedded_source(invalid_path.string());
    CHECK(invalid_source && !rag::migration::audit(**invalid_source, 1));

    const auto conflict_path = temporary.path / "conflict.ragdb";
    rag::backend::EmbeddedBackend conflict_backend(policy);
    auto conflict_document = rag::preparation::prepare_document(
        "docs/a", "conflicting alder", {{"tenant", "north"}}, "Conflict", options, &embedder);
    CHECK(conflict_document && conflict_backend.activate(*conflict_document, 1));
    CHECK(conflict_backend.checkpoint(conflict_path.string(), 0));
    auto conflict_destination = rag::migration::open_embedded_destination(conflict_path.string());
    const auto conflict = conflict_destination
                              ? rag::migration::migrate(**source, **conflict_destination,
                                                        "embedded-to-embedded-test", {1, 0})
                              : rag::fail<rag::migration::MigrationReport>(
                                    rag::Errc::unavailable, "conflict destination unavailable");
    CHECK(!conflict && conflict.error().code == rag::Errc::already_exists);

    FailingMigrationEndpoint interrupted;
    interrupted.inner = std::move(*destination);
    interrupted.writes_before_failure = 1;
    const auto first =
        rag::migration::migrate(**source, interrupted, "embedded-to-embedded-test", {1, 2});
    CHECK(!first && first.error().code == rag::Errc::io_error);
    std::filesystem::remove(destination_path.string() + ".migration");
    auto resumed_destination = rag::migration::open_embedded_destination(destination_path.string());
    CHECK(resumed_destination);
    const auto resumed = resumed_destination
                             ? rag::migration::migrate(**source, **resumed_destination,
                                                       "embedded-to-embedded-test", {1, 2})
                             : rag::fail<rag::migration::MigrationReport>(
                                   rag::Errc::unavailable, "destination did not reopen");
    CHECK(resumed && resumed->complete && resumed->resumed);
    CHECK(resumed && resumed->source_audit == resumed->destination_audit);
    CHECK(resumed && resumed->sampled_searches == 2);
    CHECK(resumed && nlohmann::json::parse(rag::migration::json_report(*resumed)).at("object") ==
                         "rag.migration_report");
    CHECK(bytes(source_path) == source_before);
    CHECK(bytes(source_path.string() + ".jobs") == jobs_before);

    const auto migrated = rag::backend::EmbeddedCheckpointStore::load(destination_path.string());
    CHECK(migrated && migrated->documents.size() == 2);
    if (migrated) {
        CHECK(migrated->documents[0].prepared->key == "docs/a");
        CHECK(migrated->documents[1].prepared->key == "docs/b");
        CHECK(migrated->documents[1].prepared->chunks[0].key == ready_document->chunks[0].key);
    }
}

void test_hnsw_checkpoint_sidecar_contract() {
    TemporaryDirectory temporary;
    const auto checkpoint = temporary.path / "hnsw.ragdb";
    rag::backend::EmbeddedMaintenancePolicy policy;
    policy.automatic_compaction = false;
    policy.dense.algorithm = rag::dense::DenseAlgorithm::hnsw;
    rag::backend::EmbeddedBackend backend(policy);
    rag::preparation::PrepareOptions options;
    options.chunking.max_lines = 1;
    options.chunking.overlap_lines = 0;
    const rag::dense::AnyEmbedder embedder{rag::dense::HashEmbedder{16}};
    auto prepared = rag::preparation::prepare_document(
        "docs/hnsw-cache", "cedar cache\nmaple cache", {}, "Cache", options, &embedder);
    CHECK(prepared && backend.activate(std::move(*prepared), 1));
    CHECK(backend.compact());
    CHECK(backend.checkpoint(checkpoint.string(), 0));
    const auto sidecar = rag::dense::hnsw_sidecar_path(checkpoint.string());
    CHECK(std::filesystem::exists(sidecar));

    auto reopened = rag::backend::EmbeddedBackend::open_checkpoint(checkpoint.string(), policy);
    CHECK(reopened);
    if (reopened) {
        const auto stats = (*reopened)->stats();
        CHECK(stats && stats->dense_algorithm == "hnsw");
        auto query = embedder.embed_one("cedar");
        CHECK(query);
        if (query) {
            rag::dense::normalize(*query);
            const auto found = (*reopened)->dense_candidates({*query, 2, {}});
            CHECK(found && found->size() == 2);
        }
    }

    std::ofstream corrupt(sidecar, std::ios::binary | std::ios::trunc);
    corrupt << "broken hnsw cache";
    corrupt.close();
    auto rebuilt = rag::backend::EmbeddedBackend::open_checkpoint(checkpoint.string(), policy);
    CHECK(rebuilt);
    CHECK(rebuilt && (*rebuilt)->stats() && (*rebuilt)->stats()->dense_algorithm == "hnsw");
    CHECK(std::filesystem::file_size(sidecar) > 17);
}

#if LRS_ENABLE_FAISS
void test_faiss_checkpoint_sidecar_contract() {
    TemporaryDirectory temporary;
    const auto checkpoint = temporary.path / "faiss.ragdb";
    rag::backend::EmbeddedMaintenancePolicy policy;
    policy.automatic_compaction = false;
    policy.dense.implementation = rag::dense::DenseImplementation::faiss;
    policy.dense.algorithm = rag::dense::DenseAlgorithm::flat;
    rag::backend::EmbeddedBackend backend(policy);
    rag::preparation::PrepareOptions options;
    options.chunking.max_lines = 1;
    options.chunking.overlap_lines = 0;
    const rag::dense::AnyEmbedder embedder{rag::dense::HashEmbedder{16}};
    auto prepared = rag::preparation::prepare_document(
        "docs/faiss-cache", "cedar cache\nmaple cache", {}, "Cache", options, &embedder);
    CHECK(prepared && backend.activate(std::move(*prepared), 1));
    CHECK(backend.compact());
    CHECK(backend.checkpoint(checkpoint.string(), 0));
    const auto sidecar =
        rag::dense::faiss_sidecar_path(checkpoint.string(), rag::dense::DenseAlgorithm::flat);
    CHECK(std::filesystem::exists(sidecar));

    auto reopened = rag::backend::EmbeddedBackend::open_checkpoint(checkpoint.string(), policy);
    CHECK(reopened);
    CHECK(reopened && (*reopened)->stats() &&
          (*reopened)->stats()->dense_implementation == "faiss");

    std::ofstream corrupt(sidecar, std::ios::binary | std::ios::trunc);
    corrupt << "broken faiss cache";
    corrupt.close();
    auto rebuilt = rag::backend::EmbeddedBackend::open_checkpoint(checkpoint.string(), policy);
    CHECK(rebuilt);
    CHECK(rebuilt && (*rebuilt)->stats() && (*rebuilt)->stats()->dense_implementation == "faiss");
    CHECK(std::filesystem::file_size(sidecar) > 18);
}
#endif

void test_durable_ingestion_coordinator() {
    TemporaryDirectory temporary;
    const auto log_path = temporary.path / "index.jobs";
    auto opened =
        rag::ingestion::AppendOnlyJobStore::open(log_path.string(), rag::store::SyncMode::none);
    CHECK(opened);
    if (!opened)
        return;
    std::shared_ptr<rag::ingestion::AppendOnlyJobStore> store = *opened;
    auto backend = std::make_shared<rag::backend::EmbeddedBackend>();
    auto gate = std::make_shared<GateState>();
    rag::preparation::PrepareOptions preparation;
    preparation.chunking.max_lines = 1;
    preparation.chunking.overlap_lines = 0;
    preparation.embedding_batch_size = 8;
    auto coordinator = rag::ingestion::IngestionCoordinator::open(
        backend, store, preparation, rag::dense::AnyEmbedder{GateEmbedder{gate}},
        rag::ingestion::CoordinatorConfig{1, 2});
    CHECK(coordinator);
    if (!coordinator)
        return;

    const auto first =
        (*coordinator)->submit({"docs/x", "X", "alpha original", {{"tenant", "north"}}}, false);
    CHECK(first && first->job.status == rag::ingestion::JobStatus::ready);
    CHECK(first && first->job.revision == 1);
    CHECK(first && first->job.prepared.has_value());
    if (first) {
        const auto view = rag::ingestion::info(first->job);
        CHECK(view.id == first->job.id);
        CHECK(view.document == "docs/x");
        CHECK(view.status == rag::ingestion::JobStatus::ready);
    }
    const auto identical =
        (*coordinator)->submit({"docs/x", "X", "alpha original", {{"tenant", "north"}}}, false);
    CHECK(identical && identical->unchanged);
    CHECK(first && identical && identical->job.id == first->job.id);

    const auto slow = (*coordinator)->submit({"docs/slow", "", "SLOW alpha", {}}, true);
    CHECK(slow && slow->job.status == rag::ingestion::JobStatus::queued);
    {
        std::unique_lock lock(gate->mutex);
        CHECK(gate->changed.wait_for(lock, std::chrono::seconds(2), [&] { return gate->entered; }));
    }
    const auto queued_one = (*coordinator)->submit({"docs/queued-1", "", "bravo queued", {}}, true);
    const auto queued_two =
        (*coordinator)->submit({"docs/queued-2", "", "charlie queued", {}}, true);
    CHECK(queued_one && queued_two);
    const auto rejected = (*coordinator)->submit({"docs/rejected", "", "delta rejected", {}}, true);
    CHECK(!rejected && rejected.error().code == rag::Errc::unavailable);
    {
        std::lock_guard lock(gate->mutex);
        gate->released = true;
    }
    gate->changed.notify_all();
    CHECK(slow && (*coordinator)->wait(slow->job.id)->status == rag::ingestion::JobStatus::ready);
    CHECK(queued_one &&
          (*coordinator)->wait(queued_one->job.id)->status == rag::ingestion::JobStatus::ready);
    CHECK(queued_two &&
          (*coordinator)->wait(queued_two->job.id)->status == rag::ingestion::JobStatus::ready);

    {
        std::lock_guard lock(gate->mutex);
        gate->entered = false;
        gate->released = false;
    }
    const auto older =
        (*coordinator)
            ->submit({"docs/x", "X2", "SLOW obsolete replacement", {{"tenant", "north"}}}, true);
    CHECK(older);
    {
        std::unique_lock lock(gate->mutex);
        CHECK(gate->changed.wait_for(lock, std::chrono::seconds(2), [&] { return gate->entered; }));
    }
    const auto newer =
        (*coordinator)->submit({"docs/x", "X3", "current cedar", {{"tenant", "north"}}}, true);
    CHECK(newer && newer->job.revision == 3);
    {
        std::lock_guard lock(gate->mutex);
        gate->released = true;
    }
    gate->changed.notify_all();
    const auto older_done = older ? (*coordinator)->wait(older->job.id)
                                  : rag::fail<rag::ingestion::IngestionJob>(rag::Errc::not_found);
    const auto newer_done = newer ? (*coordinator)->wait(newer->job.id)
                                  : rag::fail<rag::ingestion::IngestionJob>(rag::Errc::not_found);
    CHECK(older_done && older_done->status == rag::ingestion::JobStatus::superseded);
    CHECK(newer_done && newer_done->status == rag::ingestion::JobStatus::ready);
    CHECK(backend->lexical_candidates({"obsolete", 10, {}})->empty());
    CHECK(backend->lexical_candidates({"cedar", 10, {}})->size() == 1);

    const auto failed = (*coordinator)->submit({"docs/x", "X4", "FAIL replacement", {}}, false);
    CHECK(failed && failed->job.status == rag::ingestion::JobStatus::failed);
    CHECK(failed && failed->job.error &&
          failed->job.error->message.find("controlled") != std::string::npos);
    CHECK(backend->lexical_candidates({"cedar", 10, {}})->size() == 1);

    {
        std::lock_guard lock(gate->mutex);
        gate->entered = false;
        gate->released = false;
    }
    const auto pending_delete =
        (*coordinator)->submit({"docs/delete", "", "SLOW temporary", {}}, true);
    CHECK(pending_delete);
    {
        std::unique_lock lock(gate->mutex);
        CHECK(gate->changed.wait_for(lock, std::chrono::seconds(2), [&] { return gate->entered; }));
    }
    const auto deletion = (*coordinator)->erase("docs/delete");
    CHECK(deletion && deletion->operation == rag::ingestion::JobOperation::erase);
    CHECK(deletion && deletion->status == rag::ingestion::JobStatus::ready);
    {
        std::lock_guard lock(gate->mutex);
        gate->released = true;
    }
    gate->changed.notify_all();
    const auto deleted_pending =
        pending_delete ? (*coordinator)->wait(pending_delete->job.id)
                       : rag::fail<rag::ingestion::IngestionJob>(rag::Errc::not_found);
    CHECK(deleted_pending && deleted_pending->status == rag::ingestion::JobStatus::superseded);
    CHECK(backend->lexical_candidates({"temporary", 10, {}})->empty());
    const std::size_t calls_before_recovery = gate->calls.load();

    (*coordinator)->shutdown();
    coordinator =
        rag::fail<std::unique_ptr<rag::ingestion::IngestionCoordinator>>(rag::Errc::unavailable);
    store.reset();
    backend = std::make_shared<rag::backend::EmbeddedBackend>();
    auto reopened =
        rag::ingestion::AppendOnlyJobStore::open(log_path.string(), rag::store::SyncMode::none);
    CHECK(reopened);
    if (!reopened)
        return;
    store = *reopened;
    auto recovered = rag::ingestion::IngestionCoordinator::open(
        backend, store, preparation, rag::dense::AnyEmbedder{GateEmbedder{gate}},
        rag::ingestion::CoordinatorConfig{1, 2});
    CHECK(recovered);
    CHECK(gate->calls.load() == calls_before_recovery);
    CHECK(backend->lexical_candidates({"cedar", 10, {}})->size() == 1);
    CHECK(backend->lexical_candidates({"FAIL", 10, {}})->empty());
    if (recovered)
        (*recovered)->shutdown();
    recovered =
        rag::fail<std::unique_ptr<rag::ingestion::IngestionCoordinator>>(rag::Errc::unavailable);
    store.reset();

    // A torn final frame was never acknowledged and must not destroy earlier
    // ready records.
    std::ofstream torn(log_path, std::ios::binary | std::ios::app);
    torn.write("JOB", 3);
    torn.close();
    auto torn_store =
        rag::ingestion::AppendOnlyJobStore::open(log_path.string(), rag::store::SyncMode::none);
    CHECK(torn_store);
    const auto replayed =
        torn_store ? (*torn_store)->load_latest()
                   : rag::fail<std::vector<rag::ingestion::IngestionJob>>(rag::Errc::not_found);
    CHECK(replayed && !replayed->empty());
    if (torn_store) {
        rag::ingestion::IngestionJob after_torn;
        after_torn.id = "job_after_torn";
        after_torn.input.document = "docs/after-torn";
        after_torn.revision = 1;
        after_torn.content_hash = "after-torn";
        after_torn.status = rag::ingestion::JobStatus::queued;
        after_torn.created_at_ms = after_torn.updated_at_ms = 2;
        CHECK((*torn_store)->persist(after_torn));
        const auto repaired = (*torn_store)->load_latest();
        CHECK(repaired && std::any_of(repaired->begin(), repaired->end(),
                                      [](const auto& job) { return job.id == "job_after_torn"; }));
    }

    const auto prefix_path = temporary.path / "prefix.jobs";
    auto prefix_store =
        rag::ingestion::AppendOnlyJobStore::open(prefix_path.string(), rag::store::SyncMode::none);
    CHECK(prefix_store);
    if (prefix_store) {
        rag::ingestion::IngestionJob represented;
        represented.id = "job_represented";
        represented.input.document = "docs/represented";
        represented.revision = 1;
        represented.content_hash = "represented";
        represented.status = rag::ingestion::JobStatus::ready;
        represented.created_at_ms = represented.updated_at_ms = 1;
        CHECK((*prefix_store)->persist(represented));
        const auto represented_position = (*prefix_store)->size_bytes();
        rag::ingestion::IngestionJob tail = represented;
        tail.id = "job_tail";
        tail.input.document = "docs/tail";
        tail.content_hash = "tail";
        tail.created_at_ms = tail.updated_at_ms = 2;
        CHECK((*prefix_store)->persist(tail));
        const auto before_prefix = (*prefix_store)->size_bytes();
        CHECK(!(*prefix_store)->truncate_prefix(represented_position - 1));
        CHECK((*prefix_store)->size_bytes() == before_prefix);
        CHECK((*prefix_store)->truncate_prefix(represented_position, {0, 0}));
        const auto retained = (*prefix_store)->load_latest();
        CHECK(retained && retained->size() == 1);
        CHECK(retained && retained->front().id == "job_tail");
        CHECK((*prefix_store)->size_bytes() < before_prefix);
    }

    // A job interrupted in a nonterminal state is requeued after restart.
    const auto interrupted_path = temporary.path / "interrupted.jobs";
    auto interrupted_store = rag::ingestion::AppendOnlyJobStore::open(interrupted_path.string(),
                                                                      rag::store::SyncMode::none);
    CHECK(interrupted_store);
    if (!interrupted_store)
        return;
    rag::ingestion::IngestionJob interrupted;
    interrupted.id = "job_interrupted";
    interrupted.input = {"docs/recovered", "", "restart spruce", {}};
    interrupted.revision = 1;
    interrupted.content_hash =
        rag::document_content_hash("", interrupted.input.content, interrupted.input.metadata);
    interrupted.status = rag::ingestion::JobStatus::embedding;
    interrupted.created_at_ms = interrupted.updated_at_ms = 1;
    CHECK((*interrupted_store)->persist(interrupted));
    auto recovery_backend = std::make_shared<rag::backend::EmbeddedBackend>();
    auto resumed = rag::ingestion::IngestionCoordinator::open(
        recovery_backend, *interrupted_store, preparation,
        rag::dense::AnyEmbedder{GateEmbedder{gate}}, rag::ingestion::CoordinatorConfig{1, 2});
    CHECK(resumed);
    if (resumed) {
        const auto completed = (*resumed)->wait(interrupted.id);
        CHECK(completed && completed->status == rag::ingestion::JobStatus::ready);
        CHECK(recovery_backend->lexical_candidates({"spruce", 10, {}})->size() == 1);
        (*resumed)->shutdown();
    }

    const auto delete_recovery_path = temporary.path / "delete-recovery.jobs";
    auto delete_store = rag::ingestion::AppendOnlyJobStore::open(delete_recovery_path.string(),
                                                                 rag::store::SyncMode::none);
    CHECK(delete_store);
    const rag::dense::AnyEmbedder delete_embedder{GateEmbedder{gate}};
    auto prepared_delete = rag::preparation::prepare_document(
        "docs/delete-recovery", "doomed willow", {}, "", preparation, &delete_embedder);
    CHECK(prepared_delete);
    if (delete_store && prepared_delete) {
        rag::ingestion::IngestionJob ready_before_delete;
        ready_before_delete.id = "job_ready_before_delete";
        ready_before_delete.input.document = "docs/delete-recovery";
        ready_before_delete.revision = 1;
        ready_before_delete.content_hash = prepared_delete->content_hash;
        ready_before_delete.status = rag::ingestion::JobStatus::ready;
        ready_before_delete.created_at_ms = ready_before_delete.updated_at_ms = 1;
        ready_before_delete.prepared = *prepared_delete;
        CHECK((*delete_store)->persist(ready_before_delete));
        rag::ingestion::IngestionJob interrupted_delete;
        interrupted_delete.id = "job_interrupted_delete";
        interrupted_delete.operation = rag::ingestion::JobOperation::erase;
        interrupted_delete.input.document = "docs/delete-recovery";
        interrupted_delete.revision = 2;
        interrupted_delete.content_hash = "deleted";
        interrupted_delete.status = rag::ingestion::JobStatus::publishing;
        interrupted_delete.created_at_ms = interrupted_delete.updated_at_ms = 2;
        CHECK((*delete_store)->persist(interrupted_delete));
        auto deleted_backend = std::make_shared<rag::backend::EmbeddedBackend>();
        auto deletion_recovery = rag::ingestion::IngestionCoordinator::open(
            deleted_backend, *delete_store, preparation,
            rag::dense::AnyEmbedder{GateEmbedder{gate}}, rag::ingestion::CoordinatorConfig{1, 2});
        CHECK(deletion_recovery);
        CHECK(deleted_backend->lexical_candidates({"willow", 10, {}})->empty());
        if (deletion_recovery) {
            const auto recovered_delete = (*deletion_recovery)->get(interrupted_delete.id);
            CHECK(recovered_delete && recovered_delete->status == rag::ingestion::JobStatus::ready);
            (*deletion_recovery)->shutdown();
        }
    }
}

void test_composite_embedded_runtime_recovery() {
    TemporaryDirectory temporary;
    rag::ingestion::EmbeddedRuntimeConfig config;
    config.checkpoint_path = (temporary.path / "runtime.ragdb").string();
    config.job_path = (temporary.path / "runtime.jobs").string();
    config.sync_mode = rag::store::SyncMode::none;
    config.preparation.chunking.max_lines = 1;
    config.preparation.chunking.overlap_lines = 0;
    auto calls = std::make_shared<std::atomic<std::size_t>>(0);
    struct CountingEmbedder {
        std::shared_ptr<std::atomic<std::size_t>> calls;
        std::size_t dimension() const { return 16; }
        std::string_view identity() const { return "counting-hash-v1"; }
        rag::Result<std::vector<rag::Vector>> embed(std::span<const std::string> texts) const {
            ++*calls;
            return rag::dense::HashEmbedder{dimension()}.embed(texts);
        }
    };
    config.embedder = rag::dense::AnyEmbedder{CountingEmbedder{calls}};

    auto runtime = rag::ingestion::EmbeddedRuntime::open(config);
    CHECK(runtime);
    if (!runtime)
        return;
    const auto first = (*runtime)->submit(
        {"docs/runtime", "Runtime", "runtime redwood", {{"tenant", "north"}}}, false);
    CHECK(first && first->job.status == rag::ingestion::JobStatus::ready);
    const auto first_job_id = first ? first->job.id : std::string{};
    const auto calls_after_first = calls->load();
    (*runtime).reset();

    runtime = rag::ingestion::EmbeddedRuntime::open(config);
    CHECK(runtime);
    CHECK(calls->load() == calls_after_first);
    if (!runtime)
        return;
    rag::backend::SearchRequest search;
    search.query = "redwood";
    search.mode = rag::backend::SearchMode::hybrid;
    search.top_k = 2;
    search.filter = {{{"tenant", "north"}}};
    const auto recovered = (*runtime)->search(search);
    CHECK(recovered && recovered->size() == 1);
    CHECK(recovered && recovered->front().document_key == "docs/runtime");
    const auto unchanged = (*runtime)->submit(
        {"docs/runtime", "Runtime", "runtime redwood", {{"tenant", "north"}}}, false);
    CHECK(unchanged && unchanged->unchanged);

    CHECK((*runtime)->checkpoint());
    const auto checkpoint_bytes = std::filesystem::file_size(config.checkpoint_path);
    CHECK(checkpoint_bytes > 0);
    const auto second = (*runtime)->submit({"docs/tail", "", "tail sycamore", {}}, false);
    CHECK(second && second->job.status == rag::ingestion::JobStatus::ready);
    const auto calls_before_checkpoint_reopen = calls->load();
    (*runtime).reset();

    runtime = rag::ingestion::EmbeddedRuntime::open(config);
    CHECK(runtime);
    CHECK(calls->load() == calls_before_checkpoint_reopen);
    if (runtime) {
        const auto retained_job = (*runtime)->job(first_job_id);
        CHECK(retained_job && retained_job->status == rag::ingestion::JobStatus::ready);
        const auto checkpoint_result =
            (*runtime)->search({"redwood", std::nullopt, rag::backend::SearchMode::lexical, 2});
        const auto tail_result =
            (*runtime)->search({"sycamore", std::nullopt, rag::backend::SearchMode::lexical, 2});
        CHECK(checkpoint_result && checkpoint_result->size() == 1);
        CHECK(tail_result && tail_result->size() == 1);
    }

    rag::ingestion::EmbeddedRuntimeConfig automatic = config;
    automatic.checkpoint_path = (temporary.path / "automatic-runtime.ragdb").string();
    automatic.job_path = (temporary.path / "automatic-runtime.jobs").string();
    automatic.checkpoint_wal_bytes = 1;
    auto automatic_runtime = rag::ingestion::EmbeddedRuntime::open(automatic);
    CHECK(automatic_runtime);
    if (automatic_runtime) {
        const auto accepted =
            (*automatic_runtime)->submit({"docs/automatic-runtime", "", "async alder", {}}, true);
        CHECK(accepted);
        bool ready = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (accepted && std::chrono::steady_clock::now() < deadline) {
            const auto status = (*automatic_runtime)->job(accepted->job.id);
            if (status && status->status == rag::ingestion::JobStatus::ready &&
                std::filesystem::exists(automatic.checkpoint_path)) {
                ready = true;
                break;
            }
            std::this_thread::yield();
        }
        CHECK(ready);
    }
}

void test_engine_backend_neutral_entry_points() {
    TemporaryDirectory temporary;
    rag::ingestion::EmbeddedRuntimeConfig config;
    config.checkpoint_path = (temporary.path / "engine.ragdb").string();
    config.sync_mode = rag::store::SyncMode::none;
    config.maintenance.automatic_compaction = false;
    config.embedder = rag::dense::AnyEmbedder{rag::dense::HashEmbedder{16}};
    auto engine = rag::Engine::open_runtime(config);
    CHECK(engine);
    if (!engine)
        return;
    const auto indexed = engine->ingest({"docs/engine", "Engine", "engine eucalyptus", {}}, false);
    CHECK(indexed && indexed->job.status == rag::ingestion::JobStatus::ready);
    rag::backend::SearchRequest request;
    request.query = "eucalyptus";
    request.mode = rag::backend::SearchMode::hybrid;
    request.top_k = 1;
    const auto found = engine->search(request);
    CHECK(found && found->size() == 1);
    CHECK(found && found->front().document_key == "docs/engine");
    CHECK(engine->checkpoint());
    const auto removed = engine->erase("docs/engine");
    CHECK(removed && removed->status == rag::ingestion::JobStatus::ready);
}

void test_retrieval_and_persistence() {
    TemporaryDirectory temporary;
    const auto database = temporary.path / "compat.ragdb";
    rag::index::CorpusConfig config;
    config.chunk.max_lines = 1;
    config.chunk.overlap_lines = 0;
    config.hnsw_threshold = 2;
    rag::Engine engine(config);
    engine.with_embedder(rag::dense::AnyEmbedder{rag::dense::HashEmbedder{16}});
    CHECK(engine.add("a", "alpha orchard\nadjacent alpha", {{"kind", "fruit"}}));
    CHECK(engine.add("b", "beta harbor", {{"kind", "port"}}));
    CHECK(engine.build());
    CHECK(!engine.corpus().lexical_search("alpha", 5).empty());
    CHECK(engine.corpus().dense_search("alpha", 5));
    const auto fruit_filter = [](const rag::Metadata& metadata) {
        const auto found = metadata.find("kind");
        return found != metadata.end() && found->second == "fruit";
    };
    const auto filtered = engine.search("alpha", 5, fruit_filter);
    CHECK(filtered && !filtered->empty());
    CHECK(filtered && filtered->front().uri == "a");
    CHECK(filtered && filtered->front().metadata.at("kind") == "fruit");
    const auto lexical = engine.corpus().lexical_search("alpha", 5, fruit_filter);
    CHECK(!lexical.empty());
    for (const auto& hit : lexical)
        CHECK(engine.corpus().resolve(hit).uri == "a");
    const auto dense = engine.corpus().dense_search("alpha", 5, fruit_filter);
    CHECK(dense);
    CHECK(dense && !dense->empty());
    if (dense)
        for (const auto& hit : *dense)
            CHECK(engine.corpus().resolve(hit).uri == "a");
    for (const auto profile :
         {rag::retrieval::Profile::efficiency, rag::retrieval::Profile::balanced,
          rag::retrieval::Profile::quality}) {
        rag::retrieval::SearchOptions options;
        options.profile = profile;
        options.top_k = 2;
        rag::retrieval::Diagnostics diagnostics;
        const auto result = engine.search("alpha", options, &diagnostics);
        CHECK(result);
        CHECK(diagnostics.profile == profile);
        if (profile == rag::retrieval::Profile::quality) {
            bool saw_mmr = false;
            bool saw_stitch = false;
            for (const auto& stage : diagnostics.stages) {
                saw_mmr = saw_mmr || stage.find("mmr") != std::string::npos;
                saw_stitch = saw_stitch || stage.find("parent_stitch") != std::string::npos;
            }
            CHECK(saw_mmr);
            CHECK(saw_stitch);
        }
    }
    CHECK(engine.save(database.string()));
    CHECK(!engine.save(temporary.path.string()));
    CHECK(rag::Engine::open(database.string()));
    auto reopened = rag::Engine::open(database.string());
    CHECK(reopened);
    CHECK(reopened && reopened->corpus().chunks().size() == engine.corpus().chunks().size());
    if (reopened) {
        const auto id = reopened->corpus().find_by_uri("a");
        CHECK(id.has_value());
        if (id)
            CHECK(reopened->corpus().remove_document(*id));
        CHECK(reopened->save(database.string()));
    }
    auto deleted = rag::Engine::open(database.string());
    CHECK(deleted);
    CHECK(deleted && deleted->corpus().find_by_uri("a") == std::nullopt);

    std::ifstream input(database, std::ios::binary);
    std::string legacy((std::istreambuf_iterator<char>(input)), {});
    const std::uint16_t minor_zero = 0;
    std::memcpy(legacy.data() + 10, &minor_zero, sizeof(minor_zero));
    rewrite_crc(legacy);
    const auto legacy_path = temporary.path / "v1.0.ragdb";
    std::ofstream output(legacy_path, std::ios::binary);
    output.write(legacy.data(), static_cast<std::streamsize>(legacy.size()));
    output.close();
    CHECK(rag::Engine::open(legacy_path.string()));

    std::ifstream fixture_input(std::filesystem::path(LRS_SOURCE_DIR) /
                                "tests/fixtures/ragdb-v1.0-empty.hex");
    std::string fixture_hex;
    fixture_input >> fixture_hex;
    std::string fixture_bytes;
    fixture_bytes.reserve(fixture_hex.size() / 2);
    for (std::size_t index = 0; index + 1 < fixture_hex.size(); index += 2) {
        const auto byte = static_cast<char>(std::stoul(fixture_hex.substr(index, 2), nullptr, 16));
        fixture_bytes.push_back(byte);
    }
    const auto fixture_path = temporary.path / "checked-in-v1.0.ragdb";
    std::ofstream fixture_output(fixture_path, std::ios::binary);
    fixture_output.write(fixture_bytes.data(), static_cast<std::streamsize>(fixture_bytes.size()));
    fixture_output.close();
    CHECK(rag::Engine::open(fixture_path.string()));
}

void test_rrf_ordering() {
    const std::array<rag::fusion::RankedList, 2> lists{
        rag::fusion::RankedList{{rag::Hit{rag::ChunkId{2}, rag::Score{1.0F}},
                                 rag::Hit{rag::ChunkId{1}, rag::Score{0.5F}}},
                                1.0F},
        rag::fusion::RankedList{{rag::Hit{rag::ChunkId{1}, rag::Score{1.0F}},
                                 rag::Hit{rag::ChunkId{2}, rag::Score{0.5F}}},
                                1.0F}};
    const auto fused = rag::fusion::rrf(lists);
    CHECK(fused.size() == 2);
    CHECK(fused[0].chunk == rag::ChunkId{1});
}

void test_embedding_failures() {
    rag::Engine engine;
    engine.with_embedder(rag::dense::AnyEmbedder{WrongCountEmbedder{}});
    CHECK(engine.add("a", "alpha"));
    const auto built = engine.build();
    CHECK(!built);
    CHECK(built.error().code == rag::Errc::dimension_mismatch);
}

void test_container_parser() {
    TemporaryDirectory temporary;
    rag::store::Container container;
    container.put(rag::store::Tag::meta, "{}");
    container.put(rag::store::Tag::docs, std::string(4, '\0'));
    container.put_raw(0xDEADBEEFU, "future");
    auto valid = container.serialize();
    CHECK(rag::store::Container::parse(valid));
    const auto streamed_path = temporary.path / "streamed.ragdb";
    CHECK(container.write_file(streamed_path.string()));
    std::ifstream streamed_input(streamed_path, std::ios::binary);
    const std::string streamed((std::istreambuf_iterator<char>(streamed_input)),
                               std::istreambuf_iterator<char>());
    CHECK(streamed == valid);
    auto mapped = rag::store::ContainerView::open_file(streamed_path.string());
    CHECK(mapped);
    CHECK(mapped && mapped->mapped_bytes() == valid.size());
    CHECK(mapped && mapped->format_major() == rag::store::kFormatMajor);
    CHECK(mapped && mapped->format_minor() == rag::store::kFormatMinor);
    CHECK(mapped && mapped->has(rag::store::Tag::meta));
    CHECK(mapped && mapped->get(rag::store::Tag::meta) == std::optional<std::string_view>("{}"));
    CHECK(mapped && !mapped->get(rag::store::Tag::embed));
    CHECK(mapped && mapped->sections().size() == 3);
    CHECK(mapped && mapped->get_raw(0xDEADBEEFU) == std::optional<std::string_view>("future"));
    const auto owning = rag::store::Container::read_file(streamed_path.string());
    CHECK(owning && owning->serialize() == valid);
    if (mapped) {
        auto moved = std::move(*mapped);
        const auto documents = moved.get(rag::store::Tag::docs);
        CHECK(documents && documents->size() == 4);
        CHECK(documents && *documents == std::string_view("\0\0\0\0", 4));
    }
    CHECK(!rag::store::Container::parse(valid.substr(0, valid.size() - 1)));

    auto huge = valid;
    const std::uint32_t count = UINT32_MAX;
    std::memcpy(huge.data() + 16, &count, sizeof(count));
    rewrite_crc(huge);
    CHECK(!rag::store::Container::parse(huge));

    auto duplicate = valid;
    std::uint32_t first_tag = 0;
    std::memcpy(&first_tag, duplicate.data() + 28, sizeof(first_tag));
    std::memcpy(duplicate.data() + 48, &first_tag, sizeof(first_tag));
    rewrite_crc(duplicate);
    CHECK(!rag::store::Container::parse(duplicate));

    auto overlap = valid;
    std::uint64_t first_offset = 0;
    std::memcpy(&first_offset, overlap.data() + 32, sizeof(first_offset));
    std::memcpy(overlap.data() + 52, &first_offset, sizeof(first_offset));
    rewrite_crc(overlap);
    CHECK(!rag::store::Container::parse(overlap));

    const auto corrupt_path = temporary.path / "corrupt.ragdb";
    std::ofstream corrupt_output(corrupt_path, std::ios::binary);
    corrupt_output.write(overlap.data(), static_cast<std::streamsize>(overlap.size()));
    corrupt_output.close();
    const auto mapped_corrupt = rag::store::ContainerView::open_file(corrupt_path.string());
    CHECK(!mapped_corrupt);
    CHECK(!mapped_corrupt && mapped_corrupt.error().code == rag::Errc::corrupt_index);

    rag::store::Container malformed_json;
    malformed_json.put(rag::store::Tag::meta, "{");
    malformed_json.put(rag::store::Tag::docs, std::string(4, '\0'));
    malformed_json.put(rag::store::Tag::chunks, std::string(4, '\0'));
    malformed_json.put(rag::store::Tag::bm25, rag::lexical::Bm25Index{}.serialize());
    const auto malformed_path = temporary.path / "malformed-json.ragdb";
    CHECK(malformed_json.write_file(malformed_path.string()));
    CHECK(!rag::Engine::open(malformed_path.string()));

    rag::Engine vector_engine;
    vector_engine.with_embedder(rag::dense::AnyEmbedder{rag::dense::HashEmbedder{2}});
    CHECK(vector_engine.add("vector", "alpha"));
    CHECK(vector_engine.build());
    const auto vector_path = temporary.path / "valid-vector.ragdb";
    CHECK(vector_engine.save(vector_path.string()));
    auto vector_container = rag::store::Container::read_file(vector_path.string());
    CHECK(vector_container);
    if (vector_container) {
        rag::store::Container invalid_vector;
        for (const auto tag : {rag::store::Tag::meta, rag::store::Tag::docs,
                               rag::store::Tag::chunks, rag::store::Tag::bm25})
            invalid_vector.put(tag, *vector_container->get(tag));
        std::string embedding = *vector_container->get(rag::store::Tag::embed);
        const float not_a_number = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(embedding.data() + sizeof(std::uint32_t), &not_a_number, sizeof(not_a_number));
        invalid_vector.put(rag::store::Tag::embed, std::move(embedding));
        invalid_vector.set_flags(rag::store::kHasEmbeddings);
        const auto invalid_path = temporary.path / "invalid-vector.ragdb";
        CHECK(invalid_vector.write_file(invalid_path.string()));
        CHECK(!rag::Engine::open(invalid_path.string()));
    }
}

void test_wal_recovery() {
    TemporaryDirectory temporary;
    const auto path = temporary.path / "index.wal";
    {
        rag::store::Wal wal;
        CHECK(wal.open(path.string(), rag::store::SyncMode::none));
        rag::store::WalRecord record;
        record.op = rag::store::WalOp::add_document;
        record.uri = "a";
        record.text = "alpha";
        CHECK(wal.append(record));
        record.uri = "b";
        record.text = "beta";
        CHECK(wal.append(record));
    }
    std::ifstream original_input(path, std::ios::binary);
    std::string original((std::istreambuf_iterator<char>(original_input)), {});
    auto corrupt = original;
    corrupt[12] ^= 1;
    const auto corrupt_path = temporary.path / "corrupt.wal";
    std::ofstream corrupt_output(corrupt_path, std::ios::binary);
    corrupt_output.write(corrupt.data(), static_cast<std::streamsize>(corrupt.size()));
    corrupt_output.close();
    CHECK(!rag::store::Wal::replay(corrupt_path.string()));
    std::ofstream tail(path, std::ios::binary | std::ios::app);
    tail.write("WAL", 3);
    tail.close();
    const auto replayed = rag::store::Wal::replay(path.string());
    CHECK(replayed && replayed->size() == 2);
}

void test_local_http_embedder() {
    auto transport = std::make_shared<MockTransport>();
    transport->response = rag::dense::HttpResponse{
        200, R"({"data":[{"index":0,"embedding":[3,4]},{"index":1,"embedding":[0,2]}]})"};
    rag::dense::LocalHttpEmbedderConfig config;
    config.host = "127.0.0.1";
    config.dimension = 2;
    config.max_response_bytes = 1024;
    auto embedder = rag::dense::LocalHttpEmbedder::create(config, transport);
    CHECK(embedder);
    const std::array<std::string, 2> text{"a", "b"};
    const auto vectors = embedder->embed(text);
    CHECK(vectors && vectors->size() == 2);
    CHECK(vectors && std::abs((*vectors)[0][0] - 0.6F) < 0.001F);
    CHECK(transport->request.path == "/v1/embeddings");
    CHECK(transport->request.max_response_bytes == 1024);

    transport->response = rag::dense::HttpResponse{200, "{"};
    CHECK(!embedder->embed(text));
    transport->response = rag::dense::HttpResponse{
        200, R"({"data":[{"index":1,"embedding":[1,0]},{"index":0,"embedding":[0,1]}]})"};
    CHECK(!embedder->embed(text));
    transport->response =
        rag::dense::HttpResponse{200, R"({"data":[{"index":0,"embedding":[1,0]}]})"};
    CHECK(!embedder->embed(text));
    transport->response = rag::dense::HttpResponse{
        200, R"({"data":[{"index":0,"embedding":[0,0]},{"index":1,"embedding":[0,1]}]})"};
    CHECK(!embedder->embed(text));
    transport->response = rag::dense::HttpResponse{200, std::string(2048, 'x')};
    CHECK(!embedder->embed(text));
    transport->response =
        rag::fail<rag::dense::HttpResponse>(rag::Errc::transport_error, "timeout");
    CHECK(!embedder->embed(text));

    config.host = "192.0.2.1";
    CHECK(!rag::dense::LocalHttpEmbedder::create(config, transport));
    config.host = "127.0.0.1";
    config.path = "/../escape";
    CHECK(!rag::dense::LocalHttpEmbedder::create(config, transport));
}

void test_c_abi() {
    rag_engine* engine = reinterpret_cast<rag_engine*>(1);
    rag_engine_options options{};
    options.abi_version = RAG_C_ABI_VERSION;
    options.struct_size = sizeof(options);
    CHECK(rag_engine_create(nullptr, &engine) == RAG_ERR_INVALID_ARGUMENT);
    CHECK(engine == nullptr);
    CHECK(rag_engine_create(&options, &engine) == RAG_OK);
    CHECK(engine != nullptr);
    const char* key = "kind";
    CHECK(rag_engine_add(engine, "a", "alpha", nullptr, &key, nullptr, 1, nullptr) ==
          RAG_ERR_INVALID_ARGUMENT);
    rag_results* results = reinterpret_cast<rag_results*>(1);
    CHECK(rag_engine_search(engine, "alpha", 2, &key, nullptr, 1, &results) ==
          RAG_ERR_INVALID_ARGUMENT);
    CHECK(results == nullptr);
    rag_engine_free(engine);
}
} // namespace

int main() {
    test_chunking();
    test_backend_contract_types_and_exact_dense_index();
    test_tiered_dense_index_contract();
    test_native_hnsw_dense_contract();
#if LRS_ENABLE_FAISS
    test_faiss_dense_contract();
#endif
    test_embedded_backend_contract();
    test_reusable_embedded_backend_contracts();
#if LRS_ENABLE_POSTGRES
    test_postgres_backend_contract();
#endif
    test_automatic_embedded_compaction();
    test_embedded_checkpoint_contract();
    test_embedded_migration_contract();
    test_hnsw_checkpoint_sidecar_contract();
#if LRS_ENABLE_FAISS
    test_faiss_checkpoint_sidecar_contract();
#endif
    test_durable_ingestion_coordinator();
    test_composite_embedded_runtime_recovery();
    test_engine_backend_neutral_entry_points();
    test_retrieval_and_persistence();
    test_embedding_failures();
    test_rrf_ordering();
    test_container_parser();
    test_wal_recovery();
    test_local_http_embedder();
    test_c_abi();
    if (failures != 0)
        std::cerr << failures << " owned-core checks failed\n";
    return failures == 0 ? 0 : 1;
}
