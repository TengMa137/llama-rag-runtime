#include "rag/c/rag.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <vector>

#include "rag/dense/backends.hpp"
#include "rag/engine.hpp"

struct rag_engine {
    rag::Engine engine;
};
struct rag_embedder {
    std::optional<rag::dense::AnyEmbedder> embedder;
};
struct rag_results {
    std::vector<rag::SearchResult> items;
};
// Kept opaque solely so removed v1 provider/reranker entry points remain as
// binary-compatible unavailable stubs. They are intentionally not declared by
// the core-only public header.
struct rag_reranker {};

namespace {
thread_local std::string last_error;

rag_status map_error(rag::Errc error) noexcept {
    switch (error) {
        case rag::Errc::ok:
            return RAG_OK;
        case rag::Errc::not_found:
            return RAG_ERR_NOT_FOUND;
        case rag::Errc::invalid_argument:
            return RAG_ERR_INVALID_ARGUMENT;
        case rag::Errc::dimension_mismatch:
            return RAG_ERR_DIMENSION_MISMATCH;
        case rag::Errc::io_error:
            return RAG_ERR_IO;
        case rag::Errc::parse_error:
            return RAG_ERR_PARSE;
        case rag::Errc::transport_error:
            return RAG_ERR_TRANSPORT;
        case rag::Errc::unavailable:
            return RAG_ERR_UNAVAILABLE;
        case rag::Errc::empty_corpus:
            return RAG_ERR_EMPTY_CORPUS;
        case rag::Errc::already_exists:
            return RAG_ERR_ALREADY_EXISTS;
        case rag::Errc::corrupt_index:
            return RAG_ERR_CORRUPT;
    }
    return RAG_ERR_UNKNOWN;
}

rag_status from_result(const rag::Result<void>& result) {
    if (result) {
        last_error.clear();
        return RAG_OK;
    }
    last_error = result.error().message;
    return map_error(result.error().code);
}

template <class T> rag_status from_result(const rag::Result<T>& result) {
    if (result) {
        last_error.clear();
        return RAG_OK;
    }
    last_error = result.error().message;
    return map_error(result.error().code);
}

rag_status exception_status() noexcept {
    try {
        throw;
    } catch (const std::bad_alloc&) {
        last_error = "allocation failed";
    } catch (const std::exception& error) {
        last_error = error.what();
    } catch (...) {
        last_error = "unknown C++ exception";
    }
    return RAG_ERR_UNKNOWN;
}

bool valid_arrays(const char** keys, const char** values, std::size_t count) noexcept {
    return count == 0 || (keys != nullptr && values != nullptr);
}

rag::Result<rag::Metadata> make_metadata(const char** keys, const char** values,
                                         std::size_t count) {
    if (!valid_arrays(keys, values, count))
        return rag::fail<rag::Metadata>(rag::Errc::invalid_argument,
                                        "metadata arrays are required when count is non-zero");
    rag::Metadata metadata;
    for (std::size_t index = 0; index < count; ++index) {
        if (keys[index] == nullptr || values[index] == nullptr)
            return rag::fail<rag::Metadata>(rag::Errc::invalid_argument,
                                            "metadata entries must be non-null");
        metadata[keys[index]] = values[index];
    }
    return metadata;
}

} // namespace

extern "C" {

const char* rag_version(void) { return "0.2.0"; }
const char* rag_last_error(void) { return last_error.c_str(); }

rag_embedder* rag_embedder_hash(size_t dimension) {
    try {
        if (dimension == 0) {
            last_error = "embedding dimension must be non-zero";
            return nullptr;
        }
        auto result = std::make_unique<rag_embedder>();
        result->embedder.emplace(rag::dense::HashEmbedder(dimension));
        return result.release();
    } catch (...) {
        exception_status();
        return nullptr;
    }
}

rag_status rag_embedder_local_http_create(const rag_local_http_embedder_options* options,
                                          rag_embedder** out) {
    if (out != nullptr)
        *out = nullptr;
    if (options == nullptr || out == nullptr || options->abi_version != RAG_C_ABI_VERSION ||
        options->struct_size < sizeof(*options) || options->host == nullptr ||
        options->model == nullptr)
        return RAG_ERR_INVALID_ARGUMENT;
    try {
        rag::dense::LocalHttpEmbedderConfig config;
        config.host = options->host;
        config.port = options->port;
        config.path = options->path ? options->path : "/v1/embeddings";
        config.model = options->model;
        config.dimension = options->dimension;
        if (options->connect_timeout_ms != 0)
            config.connect_timeout = std::chrono::milliseconds(options->connect_timeout_ms);
        if (options->read_timeout_ms != 0)
            config.read_timeout = std::chrono::milliseconds(options->read_timeout_ms);
        if (options->max_response_bytes != 0)
            config.max_response_bytes = options->max_response_bytes;
        if (options->concurrency != 0)
            config.concurrency = options->concurrency;
        auto embedder = rag::dense::LocalHttpEmbedder::create(std::move(config));
        if (!embedder)
            return from_result(embedder);
        auto handle = std::make_unique<rag_embedder>();
        handle->embedder.emplace(std::move(*embedder));
        *out = handle.release();
        return RAG_OK;
    } catch (...) {
        return exception_status();
    }
}

void rag_embedder_free(rag_embedder* embedder) { delete embedder; }

// v1 binary-compatibility stubs for removed provider-specific factories.
rag_embedder* rag_embedder_ollama(const char*, uint16_t, const char*, size_t) { return nullptr; }
rag_embedder* rag_embedder_openai(const char*, uint16_t, int, const char*, const char*, const char*,
                                  size_t) {
    return nullptr;
}
rag_embedder* rag_embedder_llamacpp(const char*, uint16_t, size_t) { return nullptr; }
rag_reranker* rag_reranker_tei(const char*, uint16_t) { return nullptr; }
rag_reranker* rag_reranker_cohere(const char*, uint16_t, int, const char*, const char*) {
    return nullptr;
}
void rag_reranker_free(rag_reranker* reranker) { delete reranker; }

rag_engine* rag_engine_new(void) {
    try {
        return new (std::nothrow) rag_engine;
    } catch (...) {
        exception_status();
        return nullptr;
    }
}

rag_status rag_engine_create(const rag_engine_options* options, rag_engine** out) {
    if (out != nullptr)
        *out = nullptr;
    if (options == nullptr || out == nullptr || options->abi_version != RAG_C_ABI_VERSION ||
        options->struct_size < sizeof(*options) || options->writer_threads > 1)
        return RAG_ERR_INVALID_ARGUMENT;
    try {
        auto engine = std::make_unique<rag_engine>();
        *out = engine.release();
        return RAG_OK;
    } catch (...) {
        return exception_status();
    }
}

void rag_engine_free(rag_engine* engine) { delete engine; }

rag_status rag_engine_set_embedder(rag_engine* engine, rag_embedder* embedder) {
    if (engine == nullptr || embedder == nullptr || !embedder->embedder)
        return RAG_ERR_INVALID_ARGUMENT;
    try {
        engine->engine.with_embedder(*embedder->embedder);
        delete embedder;
        return RAG_OK;
    } catch (...) {
        return exception_status();
    }
}

rag_status rag_engine_set_reranker(rag_engine*, rag_reranker*, size_t, float) {
    last_error = "external reranking is outside the owned runtime core";
    return RAG_ERR_UNAVAILABLE;
}

rag_status rag_engine_add(rag_engine* engine, const char* uri, const char* text, const char* title,
                          const char** keys, const char** values, size_t count,
                          uint32_t* out_document_id) {
    if (out_document_id != nullptr)
        *out_document_id = 0;
    if (engine == nullptr || uri == nullptr || text == nullptr ||
        !valid_arrays(keys, values, count))
        return RAG_ERR_INVALID_ARGUMENT;
    try {
        auto metadata = make_metadata(keys, values, count);
        if (!metadata)
            return from_result(metadata);
        auto result = engine->engine.add(uri, text, std::move(*metadata), title ? title : "");
        if (!result)
            return from_result(result);
        if (out_document_id != nullptr)
            *out_document_id = result->get();
        return RAG_OK;
    } catch (...) {
        return exception_status();
    }
}

rag_status rag_engine_build(rag_engine* engine) {
    if (engine == nullptr)
        return RAG_ERR_INVALID_ARGUMENT;
    try {
        return from_result(engine->engine.build());
    } catch (...) {
        return exception_status();
    }
}

rag_status rag_engine_add_directory(rag_engine*, const char*, size_t* out_count) {
    if (out_count != nullptr)
        *out_count = 0;
    last_error = "directory loading is outside the owned runtime core";
    return RAG_ERR_UNAVAILABLE;
}

rag_status rag_engine_search(rag_engine* engine, const char* query, size_t count, const char** keys,
                             const char** values, size_t metadata_count, rag_results** out) {
    if (out != nullptr)
        *out = nullptr;
    if (engine == nullptr || query == nullptr || out == nullptr ||
        !valid_arrays(keys, values, metadata_count))
        return RAG_ERR_INVALID_ARGUMENT;
    try {
        rag::index::MetaFilter filter;
        if (metadata_count != 0) {
            auto metadata = make_metadata(keys, values, metadata_count);
            if (!metadata)
                return from_result(metadata);
            filter = [wanted = std::move(*metadata)](const rag::Metadata& candidate) {
                for (const auto& [key, value] : wanted) {
                    const auto found = candidate.find(key);
                    if (found == candidate.end() || found->second != value)
                        return false;
                }
                return true;
            };
        }
        auto result = engine->engine.search(query, count ? count : 10, std::move(filter));
        if (!result)
            return from_result(result);
        auto results = std::make_unique<rag_results>();
        results->items = std::move(*result);
        *out = results.release();
        return RAG_OK;
    } catch (...) {
        return exception_status();
    }
}

size_t rag_results_count(const rag_results* results) { return results ? results->items.size() : 0; }
float rag_results_score(const rag_results* results, size_t index) {
    return results && index < results->items.size() ? results->items[index].score.get() : 0.0F;
}
uint32_t rag_results_chunk_id(const rag_results* results, size_t index) {
    return results && index < results->items.size() ? results->items[index].chunk.get() : 0;
}
uint32_t rag_results_doc_id(const rag_results* results, size_t index) {
    return results && index < results->items.size() ? results->items[index].doc.get() : 0;
}
uint32_t rag_results_start_line(const rag_results* results, size_t index) {
    return results && index < results->items.size() ? results->items[index].start_line : 0;
}
uint32_t rag_results_end_line(const rag_results* results, size_t index) {
    return results && index < results->items.size() ? results->items[index].end_line : 0;
}
const char* rag_results_uri(const rag_results* results, size_t index) {
    return results && index < results->items.size() ? results->items[index].uri.c_str() : "";
}
const char* rag_results_text(const rag_results* results, size_t index) {
    return results && index < results->items.size() ? results->items[index].text.c_str() : "";
}
const char* rag_results_context(const rag_results* results, size_t index) {
    return results && index < results->items.size() ? results->items[index].context.c_str() : "";
}
void rag_results_free(rag_results* results) { delete results; }

rag_status rag_engine_save(rag_engine* engine, const char* path) {
    if (engine == nullptr || path == nullptr)
        return RAG_ERR_INVALID_ARGUMENT;
    try {
        return from_result(engine->engine.save(path));
    } catch (...) {
        return exception_status();
    }
}

rag_engine* rag_engine_load(const char* path, rag_status* out_status) {
    if (out_status != nullptr)
        *out_status = RAG_ERR_UNKNOWN;
    if (path == nullptr) {
        if (out_status != nullptr)
            *out_status = RAG_ERR_INVALID_ARGUMENT;
        return nullptr;
    }
    try {
        auto loaded = rag::Engine::open(path);
        if (!loaded) {
            last_error = loaded.error().message;
            if (out_status != nullptr)
                *out_status = map_error(loaded.error().code);
            return nullptr;
        }
        auto result = std::make_unique<rag_engine>();
        result->engine = std::move(*loaded);
        if (out_status != nullptr)
            *out_status = RAG_OK;
        return result.release();
    } catch (...) {
        const auto status = exception_status();
        if (out_status != nullptr)
            *out_status = status;
        return nullptr;
    }
}

void rag_string_free(char* string) { std::free(string); }

} // extern "C"
