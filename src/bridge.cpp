#include "lrs/bridge.h"

#include <nlohmann/json.hpp>
#include <rag/dense/backends.hpp>
#include <rag/engine.hpp>
#include <rag/pipeline/pipeline.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>

struct lrs_index {
    std::unique_ptr<rag::Engine> engine;
    std::string path;
};

namespace {
char* copy_string(const std::string& value) {
    auto* result = static_cast<char*>(std::malloc(value.size() + 1));
    if (result)
        std::memcpy(result, value.c_str(), value.size() + 1);
    return result;
}

int fail(char** error, const std::string& message) {
    if (error)
        *error = copy_string(message);
    return 1;
}

std::string normalize(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (input[i] == '\r') {
            if (i + 1 < input.size() && input[i + 1] == '\n')
                continue;
            out.push_back('\n');
        } else {
            out.push_back(input[i]);
        }
    }
    return out;
}

std::uint64_t fnv1a(std::string_view text, std::uint64_t hash = 1469598103934665603ULL) {
    for (const unsigned char c : text) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string public_chunk_id(const rag::SearchResult& hit) {
    std::string identity = hit.uri + "\n" + std::to_string(hit.start_line) + ":" +
                           std::to_string(hit.end_line) + "\n" + normalize(hit.text);
    const auto hash = fnv1a(identity);
    static constexpr char digits[] = "0123456789abcdef";
    std::string encoded(16, '0');
    for (int i = 15; i >= 0; --i)
        encoded[static_cast<std::size_t>(i)] = digits[(hash >> ((15 - i) * 4)) & 0xf];
    return "chk_" + encoded;
}

rag::index::CorpusConfig corpus_config(const lrs_index_options& options) {
    rag::index::CorpusConfig cfg;
    cfg.chunk.max_lines = 40;
    // The bundled Granite embedding model has a hard 512-token training
    // context. A byte-bounded body without an added breadcrumb is guaranteed
    // to stay below that ceiling even for byte-fallback-heavy paper text.
    cfg.chunk.max_chars = 384;
    cfg.chunk.overlap_lines = 4;
    cfg.chunk.heading_context = false;
    cfg.chunk.policy.model_identity = options.embedding_model ? options.embedding_model : "default";
    cfg.chunk.policy.dimension = options.embedding_dimension;
    if (options.measure_tokens != nullptr && options.max_input_tokens > options.reserved_tokens) {
        cfg.chunk.policy.tokenizer_identity = "backend-exact-v1";
        cfg.chunk.policy.max_tokens = options.max_input_tokens;
        cfg.chunk.policy.reserved_tokens = options.reserved_tokens;
        const auto usable = options.max_input_tokens - options.reserved_tokens;
        cfg.chunk.policy.target_tokens = std::max<std::size_t>(1, usable * 3 / 4);
        cfg.chunk.policy.overlap_tokens = usable / 8;
        cfg.chunk.policy.counting_mode = rag::text::TokenCountingMode::exact;
        const auto callback = options.measure_tokens;
        void* const context = options.token_measurer_context;
        cfg.chunk.measure_tokens = [callback, context](std::string_view text) {
            return callback(context, text.data(), text.size());
        };
    } else {
        cfg.chunk.policy.tokenizer_identity = "conservative-utf8-bytes-v1";
        cfg.chunk.policy.target_tokens = 320;
        cfg.chunk.policy.max_tokens = 384;
        cfg.chunk.policy.overlap_tokens = 32;
        cfg.chunk.policy.counting_mode = rag::text::TokenCountingMode::conservative_utf8_bytes;
    }
    cfg.chunking = rag::index::CorpusConfig::Chunking::fixed;
    return cfg;
}

void attach_embedder(rag::Engine& engine, const lrs_index_options& options) {
    if (options.deterministic_embeddings) {
        engine.with_embedder(
            rag::dense::AnyEmbedder(rag::dense::HashEmbedder(options.embedding_dimension)));
        return;
    }
    rag::dense::OpenAIConfig cfg = rag::dense::OpenAIConfig::local(
        options.embedding_host, options.embedding_port,
        options.embedding_model ? options.embedding_model : "default", options.embedding_dimension);
    engine.with_embedder(rag::dense::AnyEmbedder(rag::dense::OpenAIEmbedder(std::move(cfg))));
}

std::unique_ptr<rag::Engine> load_engine(const lrs_index_options& options) {
    std::unique_ptr<rag::Engine> engine;
    if (std::filesystem::exists(options.database_path)) {
        auto loaded = rag::Engine::open(options.database_path);
        if (!loaded)
            throw std::runtime_error(loaded.error().message);
        engine = std::make_unique<rag::Engine>(std::move(*loaded));
        const auto& stored = engine->corpus().config().chunk;
        if (stored.policy.max_tokens != 0 &&
            rag::text::chunking_fingerprint(stored) !=
                rag::text::chunking_fingerprint(corpus_config(options).chunk))
            throw std::runtime_error("embedded chunking/embedding policy is incompatible");
    } else {
        engine = std::make_unique<rag::Engine>(corpus_config(options));
    }
    attach_embedder(*engine, options);
    return engine;
}

void write_manifest(const lrs_index_options& options) {
    const auto cfg = corpus_config(options);
    nlohmann::json manifest = {{"schema", 1},
                               {"rag_cpp", "owned-v0.1.1"},
                               {"source_commit", "cfe46cee87fccb9ca5dee68d416229489285fdea"},
                               {"chunking", rag::text::chunking_fingerprint(cfg.chunk)},
                               {"max_lines", 40},
                               {"max_chars", 384},
                               {"overlap_lines", 4},
                               {"heading_context", false},
                               {"target_tokens", cfg.chunk.policy.target_tokens},
                               {"max_tokens", cfg.chunk.policy.max_tokens},
                               {"overlap_tokens", cfg.chunk.policy.overlap_tokens},
                               {"tokenizer", cfg.chunk.policy.tokenizer_identity},
                               {"model", cfg.chunk.policy.model_identity},
                               {"embedding_dimension", options.embedding_dimension}};
    const std::string final_path = std::string(options.database_path) + ".manifest.json";
    const std::string temporary = final_path + ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out)
        throw std::runtime_error("cannot write index manifest");
    out << manifest.dump(2) << '\n';
    out.close();
    std::filesystem::rename(temporary, final_path);
}

void validate_manifest(const lrs_index_options& options) {
    if (!std::filesystem::exists(options.database_path))
        return;
    const std::string path = std::string(options.database_path) + ".manifest.json";
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("index manifest is missing");
    nlohmann::json manifest;
    input >> manifest;
    const auto cfg = corpus_config(options);
    const bool legacy = manifest.value("rag_cpp", "") == "v0.1.0" &&
                        manifest.value("chunking", "") == "ragcpp-fixed-v0.1.0";
    const bool current =
        manifest.value("rag_cpp", "") == "owned-v0.1.1" &&
        manifest.value("chunking", "") == rag::text::chunking_fingerprint(cfg.chunk);
    if (manifest.value("schema", 0) != 1 || (!legacy && !current) ||
        manifest.value("embedding_dimension", 0U) != options.embedding_dimension ||
        manifest.value("max_lines", 0U) != 40U || manifest.value("max_chars", 0U) != 384U ||
        manifest.value("overlap_lines", 0U) != 4U || manifest.value("heading_context", true))
        throw std::runtime_error("index manifest is incompatible with runtime configuration");
}

int publish(std::unique_ptr<rag::Engine> next, const lrs_index_options& options,
            lrs_index** candidate, char** error) {
    const std::filesystem::path database(options.database_path);
    if (database.has_parent_path())
        std::filesystem::create_directories(database.parent_path());
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string staging = database.string() + ".generation-" + std::to_string(nonce);
    auto saved = next->save(staging);
    if (!saved)
        return fail(error, saved.error().message);
    auto validated = rag::Engine::open(staging);
    if (!validated) {
        std::filesystem::remove(staging);
        return fail(error, "candidate validation failed: " + validated.error().message);
    }
    std::filesystem::rename(staging, database);
    write_manifest(options);
    auto value = std::make_unique<lrs_index>();
    value->path = options.database_path;
    value->engine = std::move(next);
    *candidate = value.release();
    return 0;
}
} // namespace

extern "C" int lrs_index_open(const lrs_index_options* options, lrs_index** out, char** error) {
    if (!options || !out || !options->database_path || !options->embedding_host)
        return fail(error, "invalid index options");
    try {
        validate_manifest(*options);
        auto value = std::make_unique<lrs_index>();
        value->path = options->database_path;
        value->engine = load_engine(*options);
        *out = value.release();
        return 0;
    } catch (const std::exception& e) {
        return fail(error, e.what());
    }
}

extern "C" int lrs_index_stage_upsert(const lrs_index* current, const lrs_index_options* options,
                                      const char* document_id, const char* title,
                                      const char* content, lrs_index** candidate, int* unchanged,
                                      char** error) {
    if (!current || !options || !document_id || !title || !content || !candidate || !unchanged)
        return fail(error, "invalid upsert arguments");
    try {
        const std::string normalized = normalize(content);
        if (auto id = current->engine->corpus().find_by_uri(document_id)) {
            const auto* existing = current->engine->corpus().document(*id);
            if (existing && existing->text == normalized && existing->title == title) {
                *unchanged = 1;
                auto copy = std::make_unique<lrs_index>();
                copy->path = options->database_path;
                copy->engine = load_engine(*options);
                *candidate = copy.release();
                return 0;
            }
        }

        auto next = load_engine(*options);
        auto added = next->corpus().upsert_document(document_id, normalized, {}, title);
        if (!added)
            return fail(error, added.error().message);
        auto built = next->build();
        if (!built)
            return fail(error, built.error().message);

        *unchanged = 0;
        return publish(std::move(next), *options, candidate, error);
    } catch (const std::exception& e) {
        return fail(error, e.what());
    }
}

extern "C" int lrs_index_stage_delete(const lrs_index* current, const lrs_index_options* options,
                                      const char* document_id, lrs_index** candidate, int* deleted,
                                      char** error) {
    if (!current || !options || !document_id || !candidate || !deleted)
        return fail(error, "invalid delete arguments");
    try {
        auto next = load_engine(*options);
        const auto id = next->corpus().find_by_uri(document_id);
        if (!id) {
            *deleted = 0;
            auto value = std::make_unique<lrs_index>();
            value->path = options->database_path;
            value->engine = std::move(next);
            *candidate = value.release();
            return 0;
        }
        auto removed = next->corpus().remove_document(*id);
        if (!removed)
            return fail(error, removed.error().message);
        *deleted = 1;
        return publish(std::move(next), *options, candidate, error);
    } catch (const std::exception& e) {
        return fail(error, e.what());
    }
}

extern "C" int lrs_index_search_json(const lrs_index* index, const char* query, const char* mode,
                                     size_t top_k, char** json, char** error) {
    if (!index || !query || !mode || !json || top_k == 0 || top_k > 100)
        return fail(error, "invalid search arguments");
    try {
        std::vector<rag::SearchResult> results;
        const std::string selected(mode);
        if (selected == "lexical") {
            for (const auto& h : index->engine->corpus().lexical_search(query, top_k))
                results.push_back(index->engine->corpus().resolve(h));
        } else if (selected == "dense") {
            auto hits = index->engine->corpus().dense_search(query, top_k);
            if (!hits)
                return fail(error, hits.error().message);
            for (const auto& h : *hits)
                results.push_back(index->engine->corpus().resolve(h));
        } else if (selected == "hybrid") {
            rag::pipeline::HybridRetrieveConfig cfg;
            cfg.fusion = rag::pipeline::HybridRetrieveConfig::Fusion::rrf;
            auto pipeline = rag::pipeline::Pipeline::standard_with(cfg);
            auto hits = pipeline.run(index->engine->corpus(), query, top_k);
            if (!hits)
                return fail(error, hits.error().message);
            for (const auto& h : *hits)
                results.push_back(index->engine->corpus().resolve(h));
        } else
            return fail(error, "mode must be lexical, dense, or hybrid");

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
        *json = copy_string(body.dump());
        return 0;
    } catch (const std::exception& e) {
        return fail(error, e.what());
    }
}

extern "C" void lrs_index_destroy(lrs_index* index) { delete index; }
extern "C" void lrs_string_destroy(char* value) { std::free(value); }
