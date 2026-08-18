#include "lrs/mobile.h"

#include <nlohmann/json.hpp>
#include <rag/backend/embedded_backend.hpp>
#include <rag/core/keys.hpp>
#include <rag/dense/embedder.hpp>
#include <rag/engine.hpp>
#include <rag/pipeline/pipeline.hpp>
#include <rag/preparation/document_preparer.hpp>
#include <rag/retrieval/runtime.hpp>
#include <rag/text/chunker.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

struct lrs_mobile_index {
    std::string path;
    std::shared_ptr<rag::backend::EmbeddedBackend> backend;
    std::shared_ptr<rag::Engine> legacy;
    mutable std::mutex operation;
};

namespace {
char* copy_string(const std::string& value) {
    auto* result = static_cast<char*>(std::malloc(value.size() + 1));
    if (result != nullptr) {
        std::memcpy(result, value.c_str(), value.size() + 1);
    }
    return result;
}

int fail(char** error, const std::string& message) {
    if (error != nullptr) {
        *error = copy_string(message);
    }
    return 1;
}

std::string public_chunk_id(const rag::SearchResult& hit) {
    return hit.chunk_key.empty()
               ? rag::stable_chunk_key(hit.uri, hit.start_line, hit.end_line, hit.text)
               : hit.chunk_key;
}

rag::index::CorpusConfig corpus_config() {
    rag::index::CorpusConfig config;
    config.chunk.max_lines = 40;
    config.chunk.max_chars = 384;
    config.chunk.overlap_lines = 4;
    config.chunk.heading_context = false;
    config.chunk.policy.model_identity = "default";
    config.chunk.policy.tokenizer_identity = "conservative-utf8-bytes-v1";
    config.chunk.policy.target_tokens = 320;
    config.chunk.policy.max_tokens = 384;
    config.chunk.policy.overlap_tokens = 32;
    config.chunk.policy.counting_mode = rag::text::TokenCountingMode::conservative_utf8_bytes;
    return config;
}

rag::preparation::PrepareOptions preparation_options() {
    rag::preparation::PrepareOptions options;
    options.chunking = corpus_config().chunk;
    return options;
}

std::vector<rag::Chunk> chunk_document(const std::string& content) {
    return rag::text::chunk_document(rag::DocId{0}, rag::normalize_source_text(content),
                                     corpus_config().chunk);
}

rag::Vector normalized_vector(const float* values, std::size_t dimension) {
    rag::Vector result(values, values + dimension);
    double squared_norm = 0.0;
    for (const float value : result) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("embedding contains a non-finite value");
        }
        squared_norm += static_cast<double>(value) * value;
    }
    if (squared_norm <= 0.0) {
        throw std::runtime_error("embedding must have a non-zero norm");
    }
    const auto inverse_norm = static_cast<float>(1.0 / std::sqrt(squared_norm));
    for (float& value : result) {
        value *= inverse_norm;
    }
    return result;
}

class PreparedEmbedder {
  public:
    PreparedEmbedder(std::size_t dimension, std::unordered_map<std::string, rag::Vector> vectors)
        : dimension_(dimension), identity_("lrs-precomputed-f32-v1:" + std::to_string(dimension)),
          vectors_(std::move(vectors)) {}

    [[nodiscard]] std::size_t dimension() const { return dimension_; }
    [[nodiscard]] std::string_view identity() const { return identity_; }
    [[nodiscard]] std::size_t max_concurrency() const { return 1; }

    [[nodiscard]] rag::Result<std::vector<rag::Vector>>
    embed(std::span<const std::string> texts) const {
        std::vector<rag::Vector> output;
        output.reserve(texts.size());
        for (const auto& text : texts) {
            const auto found = vectors_.find(text);
            if (found == vectors_.end()) {
                return rag::fail<std::vector<rag::Vector>>(
                    rag::Errc::invalid_argument,
                    "no precomputed embedding was supplied for a requested text");
            }
            output.push_back(found->second);
        }
        return output;
    }

  private:
    std::size_t dimension_;
    std::string identity_;
    std::unordered_map<std::string, rag::Vector> vectors_;
};

std::shared_ptr<rag::Engine> load_engine(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        return std::make_shared<rag::Engine>(corpus_config());
    }
    auto loaded = rag::Engine::open(path);
    if (!loaded) {
        throw std::runtime_error(loaded.error().message);
    }
    return std::make_shared<rag::Engine>(std::move(*loaded));
}

bool has_embeddings(const rag::Engine& engine) {
    const auto chunks = engine.corpus().chunks();
    for (const auto& chunk : chunks) {
        if (!chunk.embedding.empty()) {
            return true;
        }
    }
    return false;
}

void require_vector_compatible(const rag::Engine& engine, std::size_t dimension) {
    const auto chunks = engine.corpus().chunks();
    for (const auto& chunk : chunks) {
        if (chunk.embedding.empty()) {
            throw std::runtime_error(
                "cannot add vectors to an index containing lexical-only chunks");
        }
        if (chunk.embedding.size() != dimension) {
            throw std::runtime_error("embedding dimension does not match the persisted index");
        }
    }
}

void publish(const std::string& path, const std::shared_ptr<rag::Engine>& candidate) {
    const std::filesystem::path database(path);
    if (database.has_parent_path()) {
        std::filesystem::create_directories(database.parent_path());
    }
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string staging = database.string() + ".mobile-generation-" + std::to_string(nonce);
    auto saved = candidate->save(staging);
    if (!saved) {
        throw std::runtime_error(saved.error().message);
    }
    auto validated = rag::Engine::open(staging);
    if (!validated) {
        std::filesystem::remove(staging);
        throw std::runtime_error("candidate validation failed: " + validated.error().message);
    }
    std::filesystem::rename(staging, database);
}

void publish(const std::string& path, rag::backend::EmbeddedBackend& backend) {
    const std::filesystem::path database(path);
    if (database.has_parent_path())
        std::filesystem::create_directories(database.parent_path());
    auto saved = backend.checkpoint(path, 0);
    if (!saved)
        throw std::runtime_error(saved.error().message);
}

void require_portable_mode(const rag::backend::EmbeddedBackend& backend, bool vectors,
                           std::size_t dimension = 0) {
    const auto stats = backend.stats();
    if (!stats)
        throw std::runtime_error(stats.error().message);
    if (stats->live_chunks == 0)
        return;
    if (vectors && stats->embedding_dimension == 0)
        throw std::runtime_error("cannot add vectors to an index containing lexical-only chunks");
    if (!vectors && stats->embedding_dimension != 0)
        throw std::runtime_error("cannot mix lexical-only documents into a vector index");
    if (vectors && stats->embedding_dimension != dimension)
        throw std::runtime_error("embedding dimension does not match the persisted index");
}

nlohmann::json search_results(const std::vector<rag::SearchResult>& results) {
    nlohmann::json body;
    body["results"] = nlohmann::json::array();
    std::size_t rank = 1;
    for (const auto& hit : results) {
        body["results"].push_back({{"document_id", hit.uri},
                                   {"chunk_id", public_chunk_id(hit)},
                                   {"rank", rank++},
                                   {"score", hit.score.value},
                                   {"start_line", hit.start_line},
                                   {"end_line", hit.end_line},
                                   {"text", hit.text}});
    }
    return body;
}
} // namespace

extern "C" int lrs_mobile_open(const char* database_path, lrs_mobile_index** out, char** error) {
    if (out != nullptr)
        *out = nullptr;
    if (error != nullptr)
        *error = nullptr;
    if (database_path == nullptr || database_path[0] == '\0' || out == nullptr) {
        return fail(error, "database_path and output handle are required");
    }
    try {
        auto index = std::make_unique<lrs_mobile_index>();
        index->path = database_path;
        if (std::filesystem::exists(index->path)) {
            auto portable = rag::backend::EmbeddedBackend::open_checkpoint(index->path);
            if (portable)
                index->backend =
                    std::shared_ptr<rag::backend::EmbeddedBackend>(std::move(*portable));
            else
                index->legacy = load_engine(index->path);
        } else {
            index->backend = std::make_shared<rag::backend::EmbeddedBackend>();
        }
        *out = index.release();
        return 0;
    } catch (const std::exception& exception) {
        return fail(error, exception.what());
    } catch (...) {
        return fail(error, "unknown C++ exception");
    }
}

extern "C" int lrs_mobile_prepare_document_json(const lrs_mobile_index* index,
                                                const char* document_id, const char* content,
                                                char** json, char** error) {
    if (json != nullptr)
        *json = nullptr;
    if (error != nullptr)
        *error = nullptr;
    if (index == nullptr || document_id == nullptr || content == nullptr || json == nullptr) {
        return fail(error, "index, document_id, content, and output are required");
    }
    try {
        const auto chunks = chunk_document(content);
        nlohmann::json body = {{"document_id", document_id}, {"chunks", nlohmann::json::array()}};
        std::size_t ordinal = 0;
        for (const auto& chunk : chunks) {
            body["chunks"].push_back({{"ordinal", ordinal++},
                                      {"text", chunk.text},
                                      {"embedding_text", chunk.indexed_text()},
                                      {"start_line", chunk.start_line},
                                      {"end_line", chunk.end_line}});
        }
        *json = copy_string(body.dump());
        if (*json == nullptr)
            return fail(error, "allocation failed");
        return 0;
    } catch (const std::exception& exception) {
        return fail(error, exception.what());
    } catch (...) {
        return fail(error, "unknown C++ exception");
    }
}

extern "C" int lrs_mobile_upsert_lexical(lrs_mobile_index* index, const char* document_id,
                                         const char* title, const char* content, int* unchanged,
                                         char** error) {
    if (unchanged != nullptr)
        *unchanged = 0;
    if (error != nullptr)
        *error = nullptr;
    if (index == nullptr || document_id == nullptr || title == nullptr || content == nullptr ||
        unchanged == nullptr) {
        return fail(error, "invalid lexical upsert arguments");
    }
    try {
        std::lock_guard lock(index->operation);
        if (index->backend) {
            require_portable_mode(*index->backend, false);
            auto prepared = rag::preparation::prepare_document(document_id, content, {}, title,
                                                               preparation_options(), nullptr);
            if (!prepared)
                throw std::runtime_error(prepared.error().message);
            auto state = index->backend->document_state(document_id);
            if (!state)
                throw std::runtime_error(state.error().message);
            if (*state && (*state)->content_hash == prepared->content_hash) {
                *unchanged = 1;
                return 0;
            }
            const auto revision = *state ? (*state)->revision + 1 : 1;
            auto activated = index->backend->activate(std::move(*prepared), revision);
            if (!activated)
                throw std::runtime_error(activated.error().message);
            publish(index->path, *index->backend);
            *unchanged = 0;
            return 0;
        }
        auto candidate = load_engine(index->path);
        if (has_embeddings(*candidate)) {
            throw std::runtime_error("cannot mix lexical-only documents into a vector index");
        }
        const std::string normalized = rag::normalize_source_text(content);
        if (const auto id = candidate->corpus().find_by_uri(document_id)) {
            const auto* existing = candidate->corpus().document(*id);
            if (existing != nullptr && existing->text == normalized && existing->title == title) {
                *unchanged = 1;
                return 0;
            }
        }
        auto added = candidate->corpus().upsert_document(document_id, normalized, {}, title);
        if (!added) {
            throw std::runtime_error(added.error().message);
        }
        auto built = candidate->build();
        if (!built) {
            throw std::runtime_error(built.error().message);
        }
        publish(index->path, candidate);
        index->legacy = std::move(candidate);
        *unchanged = 0;
        return 0;
    } catch (const std::exception& exception) {
        return fail(error, exception.what());
    } catch (...) {
        return fail(error, "unknown C++ exception");
    }
}

extern "C" int lrs_mobile_upsert_vectors(lrs_mobile_index* index, const char* document_id,
                                         const char* title, const char* content,
                                         const float* embeddings, std::size_t embedding_count,
                                         std::size_t embedding_dimension, int* unchanged,
                                         char** error) {
    if (unchanged != nullptr)
        *unchanged = 0;
    if (error != nullptr)
        *error = nullptr;
    if (index == nullptr || document_id == nullptr || title == nullptr || content == nullptr ||
        embeddings == nullptr || embedding_count == 0 || embedding_dimension == 0 ||
        embedding_count > SIZE_MAX / embedding_dimension || unchanged == nullptr) {
        return fail(error, "invalid vector upsert arguments");
    }
    try {
        std::lock_guard lock(index->operation);
        const std::string normalized = rag::normalize_source_text(content);
        const auto chunks = chunk_document(normalized);
        if (chunks.size() != embedding_count) {
            throw std::runtime_error("embedding count does not match prepared chunk count");
        }

        if (index->backend) {
            require_portable_mode(*index->backend, true, embedding_dimension);
            auto prepared = rag::preparation::prepare_chunks(document_id, normalized, {}, title,
                                                             preparation_options().chunking);
            if (!prepared)
                throw std::runtime_error(prepared.error().message);
            if (prepared->chunks.size() != embedding_count)
                throw std::runtime_error("embedding count does not match prepared chunk count");
            prepared->embedding_identity =
                "lrs-precomputed-f32-v1:" + std::to_string(embedding_dimension);
            for (std::size_t position = 0; position < prepared->chunks.size(); ++position)
                prepared->chunks[position].embedding = normalized_vector(
                    embeddings + position * embedding_dimension, embedding_dimension);
            auto state = index->backend->document_state(document_id);
            if (!state)
                throw std::runtime_error(state.error().message);
            if (*state && (*state)->content_hash == prepared->content_hash) {
                *unchanged = 1;
                return 0;
            }
            const auto revision = *state ? (*state)->revision + 1 : 1;
            auto activated = index->backend->activate(std::move(*prepared), revision);
            if (!activated)
                throw std::runtime_error(activated.error().message);
            publish(index->path, *index->backend);
            *unchanged = 0;
            return 0;
        }

        auto candidate = load_engine(index->path);
        require_vector_compatible(*candidate, embedding_dimension);
        if (const auto id = candidate->corpus().find_by_uri(document_id)) {
            const auto* existing = candidate->corpus().document(*id);
            if (existing != nullptr && existing->text == normalized && existing->title == title) {
                *unchanged = 1;
                return 0;
            }
        }

        std::unordered_map<std::string, rag::Vector> prepared;
        for (std::size_t position = 0; position < chunks.size(); ++position) {
            prepared[chunks[position].indexed_text()] = normalized_vector(
                embeddings + (position * embedding_dimension), embedding_dimension);
        }
        candidate->with_embedder(
            rag::dense::AnyEmbedder(PreparedEmbedder(embedding_dimension, std::move(prepared))));
        auto added = candidate->corpus().upsert_document(document_id, normalized, {}, title);
        if (!added) {
            throw std::runtime_error(added.error().message);
        }
        auto built = candidate->build();
        if (!built) {
            throw std::runtime_error(built.error().message);
        }
        publish(index->path, candidate);
        index->legacy = std::move(candidate);
        *unchanged = 0;
        return 0;
    } catch (const std::exception& exception) {
        return fail(error, exception.what());
    } catch (...) {
        return fail(error, "unknown C++ exception");
    }
}

extern "C" int lrs_mobile_search_json(lrs_mobile_index* index, const char* query,
                                      const float* query_embedding, std::size_t embedding_dimension,
                                      const char* mode, std::size_t top_k, char** json,
                                      char** error) {
    if (json != nullptr)
        *json = nullptr;
    if (error != nullptr)
        *error = nullptr;
    if (index == nullptr || query == nullptr || mode == nullptr || json == nullptr || top_k == 0 ||
        top_k > 100) {
        return fail(error, "invalid mobile search arguments");
    }
    try {
        std::lock_guard lock(index->operation);
        std::vector<rag::SearchResult> results;
        const std::string selected(mode);
        if (index->backend) {
            rag::backend::SearchRequest request;
            request.query = query;
            request.top_k = top_k;
            request.candidate_pool = std::max<std::size_t>(top_k * 6, 60);
            if (selected == "lexical") {
                request.mode = rag::backend::SearchMode::lexical;
            } else {
                if (selected != "dense" && selected != "hybrid")
                    throw std::runtime_error("mode must be lexical, dense, or hybrid");
                if (query_embedding == nullptr || embedding_dimension == 0)
                    throw std::runtime_error("dense and hybrid search require a query embedding");
                require_portable_mode(*index->backend, true, embedding_dimension);
                request.mode = selected == "dense" ? rag::backend::SearchMode::dense
                                                   : rag::backend::SearchMode::hybrid;
                request.embedding = normalized_vector(query_embedding, embedding_dimension);
            }
            auto found = rag::retrieval::search(*index->backend, request);
            if (!found)
                throw std::runtime_error(found.error().message);
            *json = copy_string(search_results(*found).dump());
            if (*json == nullptr)
                return fail(error, "allocation failed");
            return 0;
        }
        if (selected == "lexical") {
            for (const auto& hit : index->legacy->corpus().lexical_search(query, top_k)) {
                results.push_back(index->legacy->corpus().resolve(hit));
            }
        } else {
            if (selected != "dense" && selected != "hybrid") {
                throw std::runtime_error("mode must be lexical, dense, or hybrid");
            }
            if (query_embedding == nullptr || embedding_dimension == 0) {
                throw std::runtime_error("dense and hybrid search require a query embedding");
            }
            require_vector_compatible(*index->legacy, embedding_dimension);
            std::unordered_map<std::string, rag::Vector> prepared;
            prepared.emplace(query, normalized_vector(query_embedding, embedding_dimension));
            index->legacy->with_embedder(rag::dense::AnyEmbedder(
                PreparedEmbedder(embedding_dimension, std::move(prepared))));
            if (selected == "dense") {
                auto hits = index->legacy->corpus().dense_search(query, top_k);
                if (!hits) {
                    throw std::runtime_error(hits.error().message);
                }
                for (const auto& hit : *hits) {
                    results.push_back(index->legacy->corpus().resolve(hit));
                }
            } else {
                rag::pipeline::HybridRetrieveConfig config;
                config.fusion = rag::pipeline::HybridRetrieveConfig::Fusion::rrf;
                auto pipeline = rag::pipeline::Pipeline::standard_with(config);
                auto hits = pipeline.run(index->legacy->corpus(), query, top_k);
                if (!hits) {
                    throw std::runtime_error(hits.error().message);
                }
                for (const auto& hit : *hits) {
                    results.push_back(index->legacy->corpus().resolve(hit));
                }
            }
        }
        *json = copy_string(search_results(results).dump());
        if (*json == nullptr)
            return fail(error, "allocation failed");
        return 0;
    } catch (const std::exception& exception) {
        return fail(error, exception.what());
    } catch (...) {
        return fail(error, "unknown C++ exception");
    }
}

extern "C" void lrs_mobile_destroy(lrs_mobile_index* index) { delete index; }

extern "C" void lrs_mobile_string_destroy(char* value) { std::free(value); }
