#include "rag/backend/embedded_checkpoint.hpp"

#include <cmath>
#include <limits>
#include <new>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "rag/lexical/bm25.hpp"
#include "rag/store/container.hpp"
#include "rag/store/container_view.hpp"
#include "rag/store/format.hpp"

namespace rag::backend {
namespace {

using json = nlohmann::json;
constexpr std::uint32_t kRuntimeVersion = 1;

bool valid_string(std::string_view value) { return value.size() <= UINT32_MAX; }

Result<void> validate_for_save(const EmbeddedCheckpoint& checkpoint) {
    if (checkpoint.documents.size() > store::kMaxDocuments)
        return fail<void>(Errc::invalid_argument, "checkpoint document limit exceeded");
    std::size_t chunks = 0;
    std::unordered_set<DocumentKey> documents;
    std::unordered_set<ChunkKey> chunk_keys;
    for (const auto& entry : checkpoint.documents) {
        if (!entry.prepared)
            return fail<void>(Errc::invalid_argument, "checkpoint prepared document is missing");
        const auto& document = *entry.prepared;
        if (entry.revision == 0 || document.key.empty() || !documents.insert(document.key).second)
            return fail<void>(Errc::invalid_argument,
                              "checkpoint document identity or revision is invalid");
        const auto revision = checkpoint.revisions.find(document.key);
        if (revision == checkpoint.revisions.end() || revision->second != entry.revision)
            return fail<void>(Errc::invalid_argument,
                              "checkpoint active revision is missing from revision catalog");
        if (document.metadata.size() > store::kMaxMetadataEntries || !valid_string(document.key) ||
            !valid_string(document.title) || !valid_string(document.content) ||
            !valid_string(document.content_hash) || !valid_string(document.chunking_fingerprint) ||
            !valid_string(document.embedding_identity))
            return fail<void>(Errc::invalid_argument, "checkpoint document field exceeds limit");
        if (document.chunks.size() > store::kMaxChunks - chunks)
            return fail<void>(Errc::invalid_argument, "checkpoint chunk limit exceeded");
        chunks += document.chunks.size();
        for (const auto& [key, value] : document.metadata)
            if (!valid_string(key) || !valid_string(value))
                return fail<void>(Errc::invalid_argument,
                                  "checkpoint metadata field exceeds limit");
        for (const auto& chunk : document.chunks) {
            if (chunk.key.empty() || !chunk_keys.insert(chunk.key).second ||
                !valid_string(chunk.key) || !valid_string(chunk.text) ||
                !valid_string(chunk.context) || !valid_string(chunk.indexed_text) ||
                chunk.embedding.size() > store::kMaxVectorDimension)
                return fail<void>(Errc::invalid_argument, "checkpoint chunk field is invalid");
        }
    }
    if (checkpoint.revisions.size() > store::kMaxDocuments)
        return fail<void>(Errc::invalid_argument, "checkpoint revision limit exceeded");
    for (const auto& [document, revision] : checkpoint.revisions)
        if (document.empty() || !valid_string(document) || revision == 0)
            return fail<void>(Errc::invalid_argument, "checkpoint revision row is invalid");
    return {};
}

} // namespace

Result<void> EmbeddedCheckpointStore::save(const std::string& path,
                                           const EmbeddedCheckpoint& checkpoint) {
    try {
        if (auto valid = validate_for_save(checkpoint); !valid)
            return valid;

        store::Container container;
        container.put(store::Tag::meta, json{{"runtime_checkpoint", kRuntimeVersion}}.dump());

        store::Writer documents;
        documents.u<std::uint32_t>(static_cast<std::uint32_t>(checkpoint.documents.size()));
        for (std::size_t index = 0; index < checkpoint.documents.size(); ++index) {
            const auto& document = *checkpoint.documents[index].prepared;
            documents.u<std::uint32_t>(static_cast<std::uint32_t>(index));
            documents.str(document.key);
            documents.str(document.title);
            documents.str(document.content);
            documents.u<std::uint32_t>(static_cast<std::uint32_t>(document.metadata.size()));
            for (const auto& [key, value] : document.metadata) {
                documents.str(key);
                documents.str(value);
            }
        }
        container.put(store::Tag::docs, std::move(documents.data()));

        std::size_t chunk_count = 0;
        for (const auto& entry : checkpoint.documents)
            chunk_count += entry.prepared->chunks.size();
        store::Writer chunks;
        store::Writer embeddings;
        chunks.u<std::uint32_t>(static_cast<std::uint32_t>(chunk_count));
        lexical::Bm25Index lexical;
        bool any_embeddings = false;
        std::uint32_t chunk_id = 0;
        for (std::size_t document_id = 0; document_id < checkpoint.documents.size();
             ++document_id) {
            for (const auto& chunk : checkpoint.documents[document_id].prepared->chunks) {
                chunks.u<std::uint32_t>(chunk_id);
                chunks.u<std::uint32_t>(static_cast<std::uint32_t>(document_id));
                chunks.str(chunk.text);
                chunks.str(chunk.context);
                chunks.u<std::uint32_t>(chunk.start_line);
                chunks.u<std::uint32_t>(chunk.end_line);
                embeddings.u<std::uint32_t>(static_cast<std::uint32_t>(chunk.embedding.size()));
                if (!chunk.embedding.empty()) {
                    any_embeddings = true;
                    embeddings.bytes(
                        std::string_view(reinterpret_cast<const char*>(chunk.embedding.data()),
                                         chunk.embedding.size() * sizeof(float)));
                }
                lexical.add(chunk_id, chunk.indexed_text);
                ++chunk_id;
            }
        }
        lexical.finalize();
        container.put(store::Tag::chunks, std::move(chunks.data()));
        container.put(store::Tag::bm25, lexical.serialize());
        if (any_embeddings)
            container.put(store::Tag::embed, std::move(embeddings.data()));

        store::Writer runtime;
        runtime.u<std::uint32_t>(kRuntimeVersion);
        runtime.u<std::uint64_t>(checkpoint.generation);
        runtime.u<std::uint64_t>(checkpoint.wal_position);
        runtime.u<std::uint32_t>(static_cast<std::uint32_t>(checkpoint.documents.size()));
        for (const auto& entry : checkpoint.documents) {
            const auto& document = *entry.prepared;
            runtime.str(document.key);
            runtime.u<std::uint64_t>(entry.revision);
            runtime.str(document.content_hash);
            runtime.str(document.chunking_fingerprint);
            runtime.str(document.embedding_identity);
            runtime.u<std::uint32_t>(static_cast<std::uint32_t>(document.chunks.size()));
            for (const auto& chunk : document.chunks) {
                runtime.str(chunk.key);
                runtime.u<std::uint64_t>(chunk.ordinal);
                runtime.str(chunk.indexed_text);
            }
        }
        runtime.u<std::uint32_t>(static_cast<std::uint32_t>(checkpoint.revisions.size()));
        for (const auto& [document, revision] : checkpoint.revisions) {
            runtime.str(document);
            runtime.u<std::uint64_t>(revision);
        }
        container.put(store::Tag::runtime, std::move(runtime.data()));
        container.set_flags(any_embeddings ? store::kHasEmbeddings : 0);
        return container.write_file(path);
    } catch (const std::exception& error) {
        return fail<void>(Errc::io_error, "checkpoint save failed: " + std::string(error.what()));
    }
}

Result<EmbeddedCheckpoint> EmbeddedCheckpointStore::load(const std::string& path) {
    try {
        auto loaded = store::ContainerView::open_file(path);
        if (!loaded)
            return unexpected(loaded.error());
        const auto metadata = loaded->get(store::Tag::meta);
        const auto documents_blob = loaded->get(store::Tag::docs);
        const auto chunks_blob = loaded->get(store::Tag::chunks);
        const auto lexical_blob = loaded->get(store::Tag::bm25);
        const auto runtime_blob = loaded->get(store::Tag::runtime);
        if (!metadata || !documents_blob || !chunks_blob || !lexical_blob || !runtime_blob)
            return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                            "checkpoint is missing required sections");
        if (((loaded->flags() & store::kHasEmbeddings) != 0) != loaded->has(store::Tag::embed))
            return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                            "checkpoint embedding flag is inconsistent");
        const auto metadata_json = json::parse(*metadata, nullptr, false);
        if (metadata_json.is_discarded() || !metadata_json.is_object() ||
            metadata_json.value("runtime_checkpoint", 0U) != kRuntimeVersion)
            return fail<EmbeddedCheckpoint>(Errc::corrupt_index, "checkpoint metadata is invalid");

        EmbeddedCheckpoint checkpoint;
        store::Reader document_reader(*documents_blob);
        std::uint32_t document_count = 0;
        if (!document_reader.u(document_count) || document_count > store::kMaxDocuments)
            return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                            "checkpoint document count is invalid");
        checkpoint.documents.resize(document_count);
        std::vector<std::shared_ptr<PreparedDocument>> mutable_documents(document_count);
        for (std::uint32_t index = 0; index < document_count; ++index) {
            std::uint32_t id = 0;
            mutable_documents[index] = std::make_shared<PreparedDocument>();
            checkpoint.documents[index].prepared = mutable_documents[index];
            auto& document = *mutable_documents[index];
            if (!document_reader.u(id) || id != index || !document_reader.str(document.key) ||
                !document_reader.str(document.title) || !document_reader.str(document.content))
                return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                                "checkpoint document row is invalid");
            std::uint32_t metadata_count = 0;
            if (!document_reader.u(metadata_count) || metadata_count > store::kMaxMetadataEntries ||
                metadata_count > document_reader.remaining() / 8)
                return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                                "checkpoint metadata count is invalid");
            for (std::uint32_t entry = 0; entry < metadata_count; ++entry) {
                std::string key;
                std::string value;
                if (!document_reader.str(key) || !document_reader.str(value) ||
                    !document.metadata.emplace(std::move(key), std::move(value)).second)
                    return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                                    "checkpoint metadata row is invalid");
            }
        }
        if (document_reader.remaining() != 0)
            return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                            "checkpoint has trailing document data");

        struct ChunkLocation {
            std::uint32_t document = 0;
            std::size_t ordinal = 0;
        };
        std::vector<ChunkLocation> locations;
        store::Reader chunk_reader(*chunks_blob);
        std::uint32_t chunk_count = 0;
        if (!chunk_reader.u(chunk_count) || chunk_count > store::kMaxChunks)
            return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                            "checkpoint chunk count is invalid");
        locations.reserve(chunk_count);
        for (std::uint32_t index = 0; index < chunk_count; ++index) {
            std::uint32_t id = 0;
            std::uint32_t document_id = 0;
            if (!chunk_reader.u(id) || id != index || !chunk_reader.u(document_id) ||
                document_id >= checkpoint.documents.size())
                return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                                "checkpoint chunk identity is invalid");
            auto& document = *mutable_documents[document_id];
            const std::size_t ordinal = document.chunks.size();
            document.chunks.emplace_back();
            auto& chunk = document.chunks.back();
            if (!chunk_reader.str(chunk.text) || !chunk_reader.str(chunk.context) ||
                !chunk_reader.u(chunk.start_line) || !chunk_reader.u(chunk.end_line) ||
                chunk.start_line > chunk.end_line)
                return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                                "checkpoint chunk row is invalid");
            locations.push_back({document_id, ordinal});
        }
        if (chunk_reader.remaining() != 0)
            return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                            "checkpoint has trailing chunk data");

        if (const auto embedding_blob = loaded->get(store::Tag::embed)) {
            store::Reader embedding_reader(*embedding_blob);
            std::size_t dimension = 0;
            for (const auto location : locations) {
                std::uint32_t row_dimension = 0;
                if (!embedding_reader.u(row_dimension) ||
                    row_dimension > store::kMaxVectorDimension ||
                    (dimension != 0 && row_dimension != dimension))
                    return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                                    "checkpoint embedding dimension is invalid");
                if (row_dimension != 0)
                    dimension = row_dimension;
                std::string_view bytes;
                if (row_dimension > embedding_reader.remaining() / sizeof(float) ||
                    !embedding_reader.bytes(static_cast<std::size_t>(row_dimension) * sizeof(float),
                                            bytes))
                    return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                                    "checkpoint embedding row is truncated");
                auto& embedding =
                    mutable_documents[location.document]->chunks[location.ordinal].embedding;
                embedding.resize(row_dimension);
                std::memcpy(embedding.data(), bytes.data(), bytes.size());
                double norm = 0.0;
                for (const float value : embedding) {
                    if (!std::isfinite(value))
                        return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                                        "checkpoint embedding is non-finite");
                    norm += static_cast<double>(value) * value;
                }
                if (!embedding.empty() && std::abs(norm - 1.0) > 1.0e-3)
                    return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                                    "checkpoint embedding is not normalized");
            }
            if (embedding_reader.remaining() != 0)
                return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                                "checkpoint has trailing embedding data");
        }

        auto lexical = lexical::Bm25Index::deserialize(*lexical_blob);
        if (!lexical || lexical->size() != chunk_count)
            return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                            "checkpoint lexical index is invalid");

        store::Reader runtime_reader(*runtime_blob);
        std::uint32_t runtime_version = 0;
        std::uint32_t runtime_documents = 0;
        if (!runtime_reader.u(runtime_version) || runtime_version != kRuntimeVersion ||
            !runtime_reader.u(checkpoint.generation) ||
            !runtime_reader.u(checkpoint.wal_position) || !runtime_reader.u(runtime_documents) ||
            runtime_documents != document_count)
            return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                            "checkpoint runtime header is invalid");
        for (std::uint32_t index = 0; index < document_count; ++index) {
            auto& entry = checkpoint.documents[index];
            auto& document = *mutable_documents[index];
            std::string key;
            std::uint32_t runtime_chunks = 0;
            if (!runtime_reader.str(key) || key != document.key ||
                !runtime_reader.u(entry.revision) || entry.revision == 0 ||
                !runtime_reader.str(document.content_hash) ||
                !runtime_reader.str(document.chunking_fingerprint) ||
                !runtime_reader.str(document.embedding_identity) ||
                !runtime_reader.u(runtime_chunks) || runtime_chunks != document.chunks.size())
                return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                                "checkpoint runtime document is invalid");
            for (std::size_t chunk_index = 0; chunk_index < runtime_chunks; ++chunk_index) {
                auto& chunk = document.chunks[chunk_index];
                std::uint64_t ordinal = 0;
                if (!runtime_reader.str(chunk.key) || !runtime_reader.u(ordinal) ||
                    ordinal != chunk_index || !runtime_reader.str(chunk.indexed_text))
                    return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                                    "checkpoint runtime chunk is invalid");
                chunk.ordinal = static_cast<std::size_t>(ordinal);
            }
        }
        std::uint32_t revision_count = 0;
        if (!runtime_reader.u(revision_count) || revision_count > store::kMaxDocuments)
            return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                            "checkpoint revision count is invalid");
        for (std::uint32_t index = 0; index < revision_count; ++index) {
            std::string document;
            DocumentRevision revision = 0;
            if (!runtime_reader.str(document) || document.empty() || !runtime_reader.u(revision) ||
                revision == 0 ||
                !checkpoint.revisions.emplace(std::move(document), revision).second)
                return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                                "checkpoint revision row is invalid");
        }
        for (const auto& entry : checkpoint.documents) {
            const auto revision = checkpoint.revisions.find(entry.prepared->key);
            if (revision == checkpoint.revisions.end() || revision->second != entry.revision)
                return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                                "checkpoint active revision is inconsistent");
        }
        if (runtime_reader.remaining() != 0)
            return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                            "checkpoint has trailing runtime data");
        return checkpoint;
    } catch (const std::bad_alloc&) {
        return fail<EmbeddedCheckpoint>(Errc::corrupt_index, "checkpoint allocation failed");
    } catch (const std::exception& error) {
        return fail<EmbeddedCheckpoint>(Errc::corrupt_index,
                                        "checkpoint load failed: " + std::string(error.what()));
    }
}

} // namespace rag::backend
