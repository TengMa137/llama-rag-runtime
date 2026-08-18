#include "rag/backend/embedded_backend.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include "rag/backend/embedded_checkpoint.hpp"
#include "rag/core/keys.hpp"
#include "rag/dense/exact_sidecar.hpp"
#include "rag/dense/hnsw_sidecar.hpp"
#include "rag/dense/tiered_index.hpp"
#if LRS_ENABLE_FAISS
#include "rag/dense/faiss_index.hpp"
#include "rag/dense/faiss_sidecar.hpp"
#endif
#include "rag/lexical/bm25.hpp"

namespace rag::backend {

struct EmbeddedBackend::Generation {
    std::map<DocumentKey, ActiveDocument> documents;
    std::unordered_map<ChunkKey, StoredChunk> chunks;
    std::vector<ChunkKey> ordinal_keys;
    lexical::Bm25Index lexical;

    std::shared_ptr<const dense::TieredDenseIndex> dense;
    std::unordered_set<DocumentKey> delta_documents;

    std::size_t dimension = 0;
    std::string chunking_fingerprint;
    std::string embedding_identity;
    std::uint64_t number = 0;
};

EmbeddedBackend::EmbeddedBackend(EmbeddedMaintenancePolicy policy) : policy_(policy) {
    auto dense = dense::TieredDenseIndex::empty();
    if (!dense)
        return;
    auto empty = build_generation({}, std::move(*dense), {}, {}, 0);
    if (empty) {
        std::atomic_store(&active_, std::move(*empty));
        if (policy_.automatic_compaction)
            maintenance_worker_ = std::thread([this] { maintenance_loop(); });
    }
}

EmbeddedBackend::~EmbeddedBackend() {
    {
        std::lock_guard lock(maintenance_mutex_);
        stopping_ = true;
    }
    maintenance_ready_.notify_all();
    if (maintenance_worker_.joinable())
        maintenance_worker_.join();
}

Result<std::unique_ptr<EmbeddedBackend>>
EmbeddedBackend::open_checkpoint(const std::string& path, EmbeddedMaintenancePolicy policy) {
    auto checkpoint = EmbeddedCheckpointStore::load(path);
    if (!checkpoint)
        return unexpected(checkpoint.error());
    auto backend = std::make_unique<EmbeddedBackend>(policy);
    if (auto restored = backend->restore_checkpoint(*checkpoint, path); !restored)
        return unexpected(restored.error());
    return backend;
}

Result<void> EmbeddedBackend::validate(const PreparedDocument& document,
                                       DocumentRevision revision) {
    if (document.key.empty() || revision == 0)
        return fail<void>(Errc::invalid_argument, "document key and revision are required");
    if (document.content_hash.empty() || document.chunking_fingerprint.empty())
        return fail<void>(Errc::invalid_argument, "document preparation fingerprints are required");
    if (document.content != normalize_source_text(document.content) ||
        document.content_hash !=
            document_content_hash(document.title, document.content, document.metadata))
        return fail<void>(Errc::invalid_argument, "document content hash is invalid");

    std::unordered_set<ChunkKey> keys;
    std::size_t dimension = 0;
    bool any_embedding = false;
    bool any_empty = false;
    for (std::size_t index = 0; index < document.chunks.size(); ++index) {
        const auto& chunk = document.chunks[index];
        if (chunk.key.empty() || !keys.insert(chunk.key).second)
            return fail<void>(Errc::invalid_argument, "chunk keys must be non-empty and unique");
        if (chunk.ordinal != index || chunk.start_line > chunk.end_line)
            return fail<void>(Errc::invalid_argument, "chunk ordering or source lines are invalid");
        if (chunk.text.empty() || chunk.indexed_text.empty())
            return fail<void>(Errc::invalid_argument, "prepared chunk text must not be empty");
        const std::string indexed_text =
            chunk.context.empty() ? chunk.text : chunk.context + "\n" + chunk.text;
        if (chunk.indexed_text != indexed_text ||
            chunk.key !=
                stable_chunk_key(document.key, chunk.start_line, chunk.end_line, chunk.text))
            return fail<void>(Errc::invalid_argument, "prepared chunk fingerprint is invalid");
        if (chunk.embedding.empty()) {
            any_empty = true;
            continue;
        }
        any_embedding = true;
        if (dimension == 0)
            dimension = chunk.embedding.size();
        if (chunk.embedding.size() != dimension)
            return fail<void>(Errc::dimension_mismatch, "prepared embedding dimensions differ");
        double norm = 0.0;
        for (const float value : chunk.embedding) {
            if (!std::isfinite(value))
                return fail<void>(Errc::invalid_argument, "embedding contains non-finite values");
            norm += static_cast<double>(value) * static_cast<double>(value);
        }
        if (std::abs(norm - 1.0) > 1.0e-3)
            return fail<void>(Errc::invalid_argument, "prepared embedding is not normalized");
    }
    if (any_embedding && any_empty)
        return fail<void>(Errc::invalid_argument, "prepared document has partial embeddings");
    if (any_embedding && document.embedding_identity.empty())
        return fail<void>(Errc::invalid_argument, "embedding identity is required");
    return {};
}

Result<std::shared_ptr<const EmbeddedBackend::Generation>>
EmbeddedBackend::build_generation(const std::map<DocumentKey, ActiveDocument>& documents,
                                  std::shared_ptr<const dense::TieredDenseIndex> dense_index,
                                  const std::unordered_set<DocumentKey>& delta_documents,
                                  const std::unordered_set<ChunkKey>& tombstones,
                                  std::uint64_t generation_number) {
    auto generation = std::make_shared<Generation>();
    generation->documents = documents;
    generation->delta_documents = delta_documents;
    generation->number = generation_number;

    std::vector<dense::VectorRecord> delta_vectors;
    std::size_t dense_chunks = 0;
    std::size_t ordinal = 0;
    for (const auto& [document_key, active] : documents) {
        if (!active.prepared)
            return fail<std::shared_ptr<const Generation>>(Errc::invalid_argument,
                                                           "prepared document is missing");
        const auto& prepared = *active.prepared;
        if (generation->chunking_fingerprint.empty())
            generation->chunking_fingerprint = prepared.chunking_fingerprint;
        if (generation->chunking_fingerprint != prepared.chunking_fingerprint)
            return fail<std::shared_ptr<const Generation>>(
                Errc::invalid_argument, "documents use different chunking fingerprints");
        if (generation->embedding_identity.empty())
            generation->embedding_identity = prepared.embedding_identity;
        if (generation->embedding_identity != prepared.embedding_identity)
            return fail<std::shared_ptr<const Generation>>(
                Errc::invalid_argument, "documents use different embedding identities");

        for (const auto& chunk : prepared.chunks) {
            if (ordinal >= ChunkId::invalid().get())
                return fail<std::shared_ptr<const Generation>>(Errc::invalid_argument,
                                                               "chunk limit exceeded");
            if (generation->chunks.contains(chunk.key))
                return fail<std::shared_ptr<const Generation>>(Errc::already_exists,
                                                               "chunk key collision");
            generation->lexical.add(static_cast<std::uint32_t>(ordinal), chunk.indexed_text);
            generation->ordinal_keys.push_back(chunk.key);

            StoredChunk stored;
            stored.key = chunk.key;
            stored.document = document_key;
            stored.revision = active.revision;
            stored.ordinal = chunk.ordinal;
            stored.title = prepared.title;
            stored.text = chunk.text;
            stored.context = chunk.context;
            stored.metadata = prepared.metadata;
            stored.start_line = chunk.start_line;
            stored.end_line = chunk.end_line;
            generation->chunks.emplace(stored.key, std::move(stored));

            if (!chunk.embedding.empty()) {
                ++dense_chunks;
                if (generation->dimension == 0)
                    generation->dimension = chunk.embedding.size();
                if (chunk.embedding.size() != generation->dimension)
                    return fail<std::shared_ptr<const Generation>>(
                        Errc::dimension_mismatch, "documents use different embedding dimensions");
                const bool represented_by_base = dense_index->contains_base(chunk.key);
                if (generation->delta_documents.contains(document_key) && !represented_by_base) {
                    delta_vectors.push_back({chunk.key, chunk.embedding});
                }
            }
            ++ordinal;
        }
    }
    generation->lexical.finalize();
    if (dense_chunks != 0 && dense_chunks != generation->chunks.size())
        return fail<std::shared_ptr<const Generation>>(
            Errc::dimension_mismatch, "cannot mix lexical-only and dense documents");
    auto dense = dense_index->with_delta(delta_vectors, tombstones, generation->dimension);
    if (!dense)
        return unexpected(dense.error());
    generation->dense = std::move(*dense);
    return std::shared_ptr<const Generation>(std::move(generation));
}

Result<void> EmbeddedBackend::activate(PreparedDocument document, DocumentRevision revision) {
    if (auto valid = validate(document, revision); !valid)
        return valid;
    std::lock_guard lock(writer_mutex_);
    const auto latest = latest_revisions_.find(document.key);
    const auto current = std::atomic_load(&active_);
    if (latest != latest_revisions_.end() && revision <= latest->second) {
        const auto active = current->documents.find(document.key);
        if (revision == latest->second && active != current->documents.end() &&
            active->second.revision == revision &&
            active->second.prepared->content_hash == document.content_hash)
            return {};
        return fail<void>(Errc::already_exists, "document revision is stale");
    }
    auto documents = current->documents;
    auto delta_documents = current->delta_documents;
    auto tombstones = current->dense->tombstones();
    const auto old = documents.find(document.key);
    if (old != documents.end()) {
        std::unordered_set<ChunkKey> replacement_keys;
        for (const auto& chunk : document.chunks)
            replacement_keys.insert(chunk.key);
        for (const auto& chunk : old->second.prepared->chunks)
            if (current->dense->contains_base(chunk.key) && !replacement_keys.contains(chunk.key))
                tombstones.insert(chunk.key);
    }
    const auto key = document.key;
    documents[key] =
        ActiveDocument{revision, std::make_shared<const PreparedDocument>(std::move(document))};
    delta_documents.insert(key);
    auto next = build_generation(documents, current->dense, delta_documents, tombstones,
                                 current->number + 1);
    if (!next)
        return unexpected(next.error());
    latest_revisions_[key] = revision;
    std::atomic_store(&active_, std::move(*next));
    schedule_maintenance();
    return {};
}

Result<bool> EmbeddedBackend::erase(DocumentKey document, DocumentRevision revision) {
    if (document.empty() || revision == 0)
        return fail<bool>(Errc::invalid_argument, "document key and revision are required");
    std::lock_guard lock(writer_mutex_);
    const auto latest = latest_revisions_.find(document);
    const auto current = std::atomic_load(&active_);
    if (latest != latest_revisions_.end() && revision <= latest->second) {
        if (revision == latest->second && !current->documents.contains(document))
            return false;
        return fail<bool>(Errc::already_exists, "document revision is stale");
    }
    auto documents = current->documents;
    auto delta_documents = current->delta_documents;
    auto tombstones = current->dense->tombstones();
    const auto found = documents.find(document);
    const bool removed = found != documents.end();
    if (found != documents.end())
        for (const auto& chunk : found->second.prepared->chunks)
            if (current->dense->contains_base(chunk.key))
                tombstones.insert(chunk.key);
    documents.erase(document);
    delta_documents.erase(document);
    auto next = build_generation(documents, current->dense, delta_documents, tombstones,
                                 current->number + 1);
    if (!next)
        return unexpected(next.error());
    latest_revisions_[document] = revision;
    std::atomic_store(&active_, std::move(*next));
    schedule_maintenance();
    return removed;
}

Result<CandidateList> EmbeddedBackend::lexical_candidates(const LexicalRequest& request) const {
    const auto generation = std::atomic_load(&active_);
    if (!generation || request.k == 0)
        return CandidateList{};
    auto hits = generation->lexical.search(request.query, generation->ordinal_keys.size());
    CandidateList output;
    output.reserve(std::min(hits.size(), request.k));
    for (const auto& hit : hits) {
        if (hit.chunk.get() >= generation->ordinal_keys.size())
            return fail<CandidateList>(Errc::corrupt_index, "lexical ordinal is out of range");
        const auto& key = generation->ordinal_keys[hit.chunk.get()];
        const auto stored = generation->chunks.find(key);
        if (stored == generation->chunks.end())
            return fail<CandidateList>(Errc::corrupt_index, "lexical chunk is missing");
        if (request.filter && !request.filter.matches(stored->second.metadata))
            continue;
        output.push_back({key, hit.score.get(), ScoreType::bm25});
        if (output.size() == request.k)
            break;
    }
    return output;
}

Result<CandidateList> EmbeddedBackend::dense_candidates(const DenseRequest& request) const {
    const auto generation = std::atomic_load(&active_);
    if (!generation || request.k == 0)
        return CandidateList{};
    if (generation->chunks.empty())
        return CandidateList{};
    if (generation->dimension == 0)
        return fail<CandidateList>(Errc::unavailable, "embedded backend has no dense vectors");

    std::vector<ChunkKey> base_allowed;
    std::vector<ChunkKey> delta_allowed;
    const auto dense_stats = generation->dense->stats();
    base_allowed.reserve(dense_stats.base_vectors);
    delta_allowed.reserve(dense_stats.delta_vectors);
    for (const auto& [key, chunk] : generation->chunks) {
        if (request.filter && !request.filter.matches(chunk.metadata))
            continue;
        if (generation->dense->contains_base(key))
            base_allowed.push_back(key);
        if (generation->dense->contains_delta(key))
            delta_allowed.push_back(key);
    }
    return generation->dense->search(request.query, base_allowed, delta_allowed, request.k);
}

Result<std::vector<StoredChunk>> EmbeddedBackend::fetch(std::span<const ChunkKey> chunks,
                                                        FetchOptions options) const {
    const auto generation = std::atomic_load(&active_);
    if (!generation)
        return fail<std::vector<StoredChunk>>(Errc::unavailable,
                                              "embedded backend is not initialized");
    std::vector<StoredChunk> output;
    output.reserve(chunks.size());
    for (const auto& key : chunks) {
        const auto found = generation->chunks.find(key);
        if (found == generation->chunks.end())
            continue;
        output.push_back(found->second);
        if (!options.include_text) {
            output.back().text.clear();
            output.back().context.clear();
        }
        if (options.include_embedding) {
            const auto document = generation->documents.find(found->second.document);
            if (document == generation->documents.end() ||
                found->second.ordinal >= document->second.prepared->chunks.size())
                return fail<std::vector<StoredChunk>>(Errc::corrupt_index,
                                                      "chunk embedding source is missing");
            const auto& prepared = document->second.prepared->chunks[found->second.ordinal];
            if (prepared.key != key)
                return fail<std::vector<StoredChunk>>(
                    Errc::corrupt_index, "chunk embedding source does not match catalog");
            output.back().embedding = prepared.embedding;
        }
    }
    return output;
}

Result<BackendStats> EmbeddedBackend::stats() const {
    const auto generation = std::atomic_load(&active_);
    if (!generation)
        return fail<BackendStats>(Errc::unavailable, "embedded backend is not initialized");
    BackendStats output;
    output.live_documents = generation->documents.size();
    output.live_chunks = generation->chunks.size();
    output.embedding_dimension = generation->dimension;
    const auto dense_stats = generation->dense->stats();
    output.dense_base_chunks = dense_stats.base_vectors;
    output.dense_delta_chunks = dense_stats.delta_vectors;
    output.tombstones = dense_stats.tombstones;
    output.generation = generation->number;
    output.dense_bytes = dense_stats.resident_bytes;
    output.dense_mapped_bytes = dense_stats.mapped_bytes;
    output.dense_exact = dense_stats.base_exact;
    output.dense_implementation = dense_stats.base_implementation;
    output.dense_algorithm = dense_stats.base_algorithm;
    output.estimated_compaction_bytes =
        generation->chunks.size() * generation->dimension * sizeof(float);
    const bool delta_limit = output.dense_delta_chunks >= policy_.delta_chunk_limit;
    const bool delta_fraction =
        output.dense_base_chunks != 0 &&
        static_cast<double>(output.dense_delta_chunks) >=
            static_cast<double>(output.dense_base_chunks) * policy_.delta_base_fraction;
    const bool tombstone_fraction =
        output.dense_base_chunks != 0 &&
        static_cast<double>(output.tombstones) >=
            static_cast<double>(output.dense_base_chunks) * policy_.tombstone_base_fraction;
    const bool compaction_over_budget =
        output.estimated_compaction_bytes > policy_.compaction_memory_budget;
    const bool automatic_base_due =
        output.dense_base_chunks == 0 && output.dense_delta_chunks >= policy_.dense.exact_threshold;
    output.maintenance_required = delta_limit || delta_fraction || tombstone_fraction ||
                                  compaction_over_budget || automatic_base_due;
    output.capabilities = {true, generation->dimension != 0, false, true, true};
    return output;
}

Result<std::optional<ActiveDocumentState>>
EmbeddedBackend::document_state(const DocumentKey& document) const {
    const auto generation = std::atomic_load(&active_);
    if (!generation)
        return fail<std::optional<ActiveDocumentState>>(Errc::unavailable,
                                                        "embedded backend is not initialized");
    const auto found = generation->documents.find(document);
    if (found == generation->documents.end())
        return std::optional<ActiveDocumentState>{};
    return std::optional<ActiveDocumentState>{
        ActiveDocumentState{found->second.revision, found->second.prepared->content_hash}};
}

Result<void> EmbeddedBackend::compact(bool override_memory_budget) {
    const auto snapshot = std::atomic_load(&active_);
    const std::size_t estimated = snapshot->chunks.size() * snapshot->dimension * sizeof(float);
    if (!override_memory_budget && estimated > policy_.compaction_memory_budget)
        return fail<void>(Errc::unavailable, "compaction exceeds configured memory budget");

    std::vector<dense::VectorRecord> vectors;
    vectors.reserve(snapshot->chunks.size());
    for (const auto& [document_key, active] : snapshot->documents)
        for (const auto& chunk : active.prepared->chunks) {
            if (chunk.embedding.empty())
                continue;
            vectors.push_back({chunk.key, chunk.embedding});
        }
    auto dense = snapshot->dense->compact(vectors, snapshot->dimension, policy_.dense);
    if (!dense)
        return unexpected(dense.error());
    auto compacted =
        build_generation(snapshot->documents, std::move(*dense), {}, {}, snapshot->number + 1);
    if (!compacted)
        return unexpected(compacted.error());

    std::lock_guard lock(writer_mutex_);
    if (std::atomic_load(&active_) != snapshot)
        return fail<void>(Errc::unavailable, "generation changed while compaction was building");
    std::atomic_store(&active_, std::move(*compacted));
    return {};
}

Result<void> EmbeddedBackend::checkpoint(const std::string& path, std::uint64_t wal_position) {
    EmbeddedCheckpoint checkpoint;
    std::shared_ptr<const Generation> snapshot;
    {
        std::lock_guard lock(writer_mutex_);
        snapshot = std::atomic_load(&active_);
        if (!snapshot)
            return fail<void>(Errc::unavailable, "embedded backend is not initialized");
        checkpoint.generation = snapshot->number;
        checkpoint.wal_position = wal_position;
        checkpoint.revisions = latest_revisions_;
    }
    checkpoint.documents.reserve(snapshot->documents.size());
    for (const auto& [key, active] : snapshot->documents)
        checkpoint.documents.push_back({active.prepared, active.revision});
    if (auto saved = EmbeddedCheckpointStore::save(path, checkpoint); !saved)
        return saved;
    std::vector<dense::VectorRecord> vectors;
    vectors.reserve(snapshot->chunks.size());
    for (const auto& [document_key, active] : snapshot->documents)
        for (const auto& chunk : active.prepared->chunks)
            if (!chunk.embedding.empty())
                vectors.push_back({chunk.key, chunk.embedding});
    const auto algorithm = dense::resolve_dense_algorithm(policy_.dense, vectors.size());
    if (algorithm && policy_.dense.implementation == dense::DenseImplementation::faiss) {
#if LRS_ENABLE_FAISS
        const auto dense_stats = snapshot->dense->stats();
        if (dense_stats.delta_vectors == 0 && dense_stats.tombstones == 0) {
            const auto faiss =
                std::dynamic_pointer_cast<const dense::FaissIndex>(snapshot->dense->base_index());
            if (faiss) {
                const auto fingerprint = dense::faiss_sidecar_fingerprint(
                    vectors, snapshot->embedding_identity, *algorithm, policy_.dense.faiss);
                (void)dense::write_faiss_sidecar(dense::faiss_sidecar_path(path, *algorithm),
                                                 fingerprint, *faiss);
            }
        }
#endif
    } else if (algorithm && *algorithm == dense::DenseAlgorithm::exact) {
        const auto fingerprint =
            dense::exact_sidecar_fingerprint(vectors, snapshot->embedding_identity);
        (void)dense::write_exact_sidecar(dense::exact_sidecar_path(path), fingerprint, vectors);
    } else if (algorithm && *algorithm == dense::DenseAlgorithm::hnsw) {
        const auto dense_stats = snapshot->dense->stats();
        if (dense_stats.delta_vectors == 0 && dense_stats.tombstones == 0) {
            const auto hnsw = std::dynamic_pointer_cast<const dense::NativeHnswIndex>(
                snapshot->dense->base_index());
            if (hnsw) {
                const auto fingerprint = dense::hnsw_sidecar_fingerprint(
                    vectors, snapshot->embedding_identity, policy_.dense.hnsw);
                (void)dense::write_hnsw_sidecar(dense::hnsw_sidecar_path(path), fingerprint, *hnsw);
            }
        }
    }
    checkpoint_wal_position_.store(wal_position);
    return {};
}

Result<void> EmbeddedBackend::restore_checkpoint(const EmbeddedCheckpoint& checkpoint,
                                                 const std::string& checkpoint_path) {
    std::map<DocumentKey, ActiveDocument> documents;
    auto revisions = checkpoint.revisions;
    for (const auto& [document, revision] : revisions)
        if (document.empty() || revision == 0)
            return fail<void>(Errc::corrupt_index, "checkpoint revision catalog is invalid");
    std::vector<dense::VectorRecord> vectors;
    std::vector<ChunkKey> vector_keys;
    std::size_t dimension = 0;
    std::string embedding_identity;
    for (const auto& entry : checkpoint.documents) {
        if (!entry.prepared)
            return fail<void>(Errc::corrupt_index, "checkpoint prepared document is missing");
        const auto& prepared = *entry.prepared;
        if (auto valid = validate(prepared, entry.revision); !valid)
            return fail<void>(Errc::corrupt_index,
                              "checkpoint prepared document is invalid: " + valid.error().message);
        if (!documents.emplace(prepared.key, ActiveDocument{entry.revision, entry.prepared}).second)
            return fail<void>(Errc::corrupt_index, "checkpoint has duplicate document keys");
        const auto catalog_revision = revisions.find(prepared.key);
        if (catalog_revision == revisions.end() || catalog_revision->second != entry.revision)
            return fail<void>(Errc::corrupt_index,
                              "checkpoint active revision does not match catalog");
        for (const auto& chunk : prepared.chunks) {
            if (chunk.embedding.empty())
                continue;
            if (dimension == 0)
                dimension = chunk.embedding.size();
            vectors.push_back({chunk.key, chunk.embedding});
            vector_keys.push_back(chunk.key);
        }
        if (!prepared.embedding_identity.empty())
            embedding_identity = prepared.embedding_identity;
    }
    const auto algorithm = dense::resolve_dense_algorithm(policy_.dense, vectors.size());
    if (!algorithm)
        return unexpected(algorithm.error());

    Result<std::shared_ptr<const dense::TieredDenseIndex>> dense_index =
        fail<std::shared_ptr<const dense::TieredDenseIndex>>(Errc::unavailable,
                                                             "dense base is unavailable");
    if (policy_.dense.implementation == dense::DenseImplementation::faiss) {
#if LRS_ENABLE_FAISS
        const auto sidecar = dense::faiss_sidecar_path(checkpoint_path, *algorithm);
        const auto fingerprint = dense::faiss_sidecar_fingerprint(vectors, embedding_identity,
                                                                  *algorithm, policy_.dense.faiss);
        auto faiss = dense::load_faiss_sidecar(sidecar, fingerprint, vector_keys, *algorithm,
                                               policy_.dense.faiss);
        if (faiss) {
            dense_index = dense::TieredDenseIndex::from_base(*faiss, vector_keys, dimension);
        } else {
            auto empty = dense::TieredDenseIndex::empty();
            if (!empty)
                return unexpected(empty.error());
            dense_index = (*empty)->compact(vectors, dimension, policy_.dense);
            if (dense_index) {
                const auto built_faiss = std::dynamic_pointer_cast<const dense::FaissIndex>(
                    (*dense_index)->base_index());
                if (built_faiss)
                    (void)dense::write_faiss_sidecar(sidecar, fingerprint, *built_faiss);
            }
        }
#else
        return fail<void>(Errc::unavailable, "FAISS support is not enabled in this build");
#endif
    } else if (*algorithm == dense::DenseAlgorithm::exact) {
        const auto sidecar = dense::exact_sidecar_path(checkpoint_path);
        const auto fingerprint = dense::exact_sidecar_fingerprint(vectors, embedding_identity);
        auto mapped = dense::MappedExactIndex::open(sidecar, fingerprint);
        if (!mapped) {
            if (auto written = dense::write_exact_sidecar(sidecar, fingerprint, vectors); written)
                mapped = dense::MappedExactIndex::open(sidecar, fingerprint);
        }
        if (mapped)
            dense_index = dense::TieredDenseIndex::from_base(*mapped, vector_keys, dimension);
        else {
            auto empty = dense::TieredDenseIndex::empty();
            if (!empty)
                return unexpected(empty.error());
            dense_index = (*empty)->compact(vectors, dimension, policy_.dense);
        }
    } else {
        const auto fingerprint =
            dense::hnsw_sidecar_fingerprint(vectors, embedding_identity, policy_.dense.hnsw);
        auto hnsw = dense::load_hnsw_sidecar(dense::hnsw_sidecar_path(checkpoint_path), fingerprint,
                                             vector_keys, policy_.dense.hnsw);
        if (hnsw) {
            dense_index = dense::TieredDenseIndex::from_base(*hnsw, vector_keys, dimension);
        } else {
            auto empty = dense::TieredDenseIndex::empty();
            if (!empty)
                return unexpected(empty.error());
            dense_index = (*empty)->compact(vectors, dimension, policy_.dense);
            if (dense_index) {
                const auto built_hnsw = std::dynamic_pointer_cast<const dense::NativeHnswIndex>(
                    (*dense_index)->base_index());
                if (built_hnsw)
                    (void)dense::write_hnsw_sidecar(dense::hnsw_sidecar_path(checkpoint_path),
                                                    fingerprint, *built_hnsw);
            }
        }
    }
    if (!dense_index)
        return unexpected(dense_index.error());
    auto restored =
        build_generation(documents, std::move(*dense_index), {}, {}, checkpoint.generation);
    if (!restored)
        return fail<void>(Errc::corrupt_index,
                          "checkpoint generation is invalid: " + restored.error().message);

    std::lock_guard lock(writer_mutex_);
    latest_revisions_ = std::move(revisions);
    std::atomic_store(&active_, std::move(*restored));
    checkpoint_wal_position_.store(checkpoint.wal_position);
    return {};
}

void EmbeddedBackend::schedule_maintenance() {
    if (!policy_.automatic_compaction)
        return;
    {
        std::lock_guard lock(maintenance_mutex_);
        maintenance_pending_ = true;
    }
    maintenance_ready_.notify_one();
}

void EmbeddedBackend::maintenance_loop() {
    while (true) {
        {
            std::unique_lock lock(maintenance_mutex_);
            maintenance_ready_.wait(lock, [this] { return stopping_ || maintenance_pending_; });
            if (stopping_)
                return;
            maintenance_pending_ = false;
        }
        const auto current = stats();
        if (current && current->maintenance_required)
            (void)compact();
    }
}

} // namespace rag::backend
