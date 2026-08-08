#pragma once
// Owned embedding implementations: deterministic local hashing and a tightly
// constrained HTTP client for a coordinator-managed loopback model process.

#include <chrono>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/dense/embedder.hpp"

namespace rag::dense {

struct LocalHttpEmbedderConfig {
    std::string host = "127.0.0.1";
    std::uint16_t port = 8080;
    std::string path = "/v1/embeddings";
    std::string model = "default";
    std::size_t dimension = 0;
    std::chrono::milliseconds connect_timeout{5'000};
    std::chrono::milliseconds read_timeout{30'000};
    std::size_t max_response_bytes = 16 * 1024 * 1024;
    std::size_t concurrency = 4;
};

class LocalHttpEmbedder {
  public:
    [[nodiscard]] static Result<LocalHttpEmbedder>
    create(LocalHttpEmbedderConfig cfg,
           std::shared_ptr<HttpTransport> transport = default_http_transport());

    [[nodiscard]] std::size_t dimension() const noexcept { return cfg_.dimension; }
    [[nodiscard]] std::string_view identity() const noexcept { return cfg_.model; }
    [[nodiscard]] std::size_t max_concurrency() const noexcept { return cfg_.concurrency; }
    [[nodiscard]] Result<std::vector<Vector>> embed(std::span<const std::string> texts) const;

  private:
    LocalHttpEmbedder(LocalHttpEmbedderConfig cfg, std::shared_ptr<HttpTransport> transport)
        : cfg_(std::move(cfg)), transport_(std::move(transport)) {}

    LocalHttpEmbedderConfig cfg_;
    std::shared_ptr<HttpTransport> transport_;
};

class HashEmbedder {
  public:
    explicit HashEmbedder(std::size_t dimension = 256) : dimension_(dimension) {}
    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
    [[nodiscard]] std::string_view identity() const noexcept { return "hash-embed-v1"; }
    [[nodiscard]] std::size_t max_concurrency() const noexcept {
        const unsigned count = std::thread::hardware_concurrency();
        return count ? count : 1u;
    }
    [[nodiscard]] Result<std::vector<Vector>> embed(std::span<const std::string> texts) const;

  private:
    std::size_t dimension_;
};

static_assert(Embedder<LocalHttpEmbedder>);
static_assert(Embedder<HashEmbedder>);

} // namespace rag::dense
