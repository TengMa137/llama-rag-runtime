#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace lrs {
struct Endpoint {
    std::string host = "127.0.0.1";
    std::uint16_t port = 0;
};

// Parent-runtime configuration contract. These are plain transport values;
// conversion to rag.cpp policy enums happens at the service boundary.
struct NativeHnswConfig {
    std::size_t neighbors = 16;
    std::size_t ef_construction = 200;
    std::size_t ef_search = 64;
    std::uint64_t seed = 0x9E3779B97F4A7C15ULL;
};

struct FaissDenseConfig {
    std::size_t hnsw_neighbors = 32;
    std::size_t ef_construction = 200;
    std::size_t ef_search = 64;
    std::size_t ivf_lists = 256;
    std::size_t ivf_probes = 16;
    std::size_t minimum_training_vectors_per_list = 39;
    std::size_t pq_subquantizers = 32;
    std::size_t pq_bits = 8;
};

struct DenseConfig {
    std::string implementation = "native";
    std::string algorithm = "automatic";
    std::size_t exact_threshold = 2'000;
    NativeHnswConfig hnsw;
    FaissDenseConfig faiss;
};

struct EmbeddedRetrievalConfig {
    std::string path = "data/knowledge.ragdb";
    DenseConfig dense;
};

struct RetrievalConfig {
    std::string backend = "embedded";
    EmbeddedRetrievalConfig embedded;
    struct Postgres {
        std::string connection_env = "LRS_POSTGRES_URL";
        std::string schema = "lrs_rag";
        std::string corpus = "default";
        std::size_t pool_size = 4;
        std::size_t acquire_timeout_ms = 5'000;
        std::size_t statement_timeout_ms = 10'000;
        std::string vector_index = "exact";
        std::size_t hnsw_ef_search = 64;
    } postgres;
};

struct Config {
    Endpoint listen{"127.0.0.1", 8080};
    Endpoint embedding{"127.0.0.1", 8081};
    Endpoint generation{"127.0.0.1", 8082};
    std::string index_path = "data/knowledge.ragdb";
    RetrievalConfig retrieval;
    std::string llama_server = "llama-server";
    std::string embedding_model;
    std::string generation_model;
    std::string embedding_api_model = "default";
    std::string generation_api_model = "default";
    std::string pooling = "mean";
    std::size_t embedding_dimension = 768;
    std::size_t embedding_context_size = 512;
    std::size_t embedding_batch_size = 2048;
    std::size_t generation_context_size = 16384;
    std::size_t context_tokens = 8192;
    std::string api_key;
    bool spawn = true;
    bool deterministic_embeddings = false;
    std::size_t ingestion_workers = 1;
    std::size_t ingestion_queue_capacity = 64;
};

Config load_config(const std::string& path);
void validate_config(const Config& config);
[[nodiscard]] bool authenticate_api_key(const Config& config, std::string_view supplied) noexcept;
[[nodiscard]] bool authorize_api_request(const Config& config, std::size_t header_count,
                                         std::string_view supplied) noexcept;
} // namespace lrs
