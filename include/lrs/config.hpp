#pragma once

#include <cstdint>
#include <string>

namespace lrs {
struct Endpoint {
    std::string host = "127.0.0.1";
    std::uint16_t port = 0;
};
struct Config {
    Endpoint listen{"127.0.0.1", 8080};
    Endpoint embedding{"127.0.0.1", 8081};
    Endpoint generation{"127.0.0.1", 8082};
    std::string index_path = "data/knowledge.ragdb";
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
    bool spawn = true;
    bool deterministic_embeddings = false;
};

Config load_config(const std::string& path);
void validate_config(const Config& config);
} // namespace lrs
