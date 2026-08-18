#pragma once
// Replaceable dense candidate index. Durable document state lives elsewhere.

#include <cstddef>
#include <shared_mutex>
#include <span>
#include <string>
#include <vector>

#include "rag/backend/candidate_backend.hpp"

namespace rag::dense {

struct VectorRecord {
    std::string_view key;
    VectorView vector;
};

// Non-owning build input. Keys and vector rows must remain alive only for the
// duration of the call; every DenseIndex implementation copies or serializes
// what it needs before returning.
using VectorSource = std::span<const VectorRecord>;

// An empty allow-list means all rows are allowed. Non-empty lists are exact.
struct AllowedIds {
    std::span<const backend::ChunkKey> ids;

    [[nodiscard]] bool allows(const backend::ChunkKey& key) const noexcept;
};

struct DenseIndexStats {
    std::size_t vectors = 0;
    std::size_t dimension = 0;
    std::size_t resident_bytes = 0;
    std::size_t primary_vector_bytes = 0;
    std::size_t compressed_vector_bytes = 0;
    bool exact = false;
    std::string implementation;
    std::string algorithm;
    std::size_t mapped_bytes = 0;
};

class DenseIndex {
  public:
    virtual ~DenseIndex() = default;
    virtual Result<void> build(VectorSource source) = 0;
    [[nodiscard]] virtual Result<backend::CandidateList> search(Vector query, AllowedIds allowed,
                                                                std::size_t k) const = 0;
    [[nodiscard]] virtual DenseIndexStats stats() const = 0;
};

// Correctness oracle and mutable-delta implementation. It owns one normalized
// f32 copy and performs exact inner-product cosine search.
class NativeExactIndex final : public DenseIndex {
  public:
    Result<void> build(VectorSource source) override;
    [[nodiscard]] Result<backend::CandidateList> search(Vector query, AllowedIds allowed,
                                                        std::size_t k) const override;
    [[nodiscard]] DenseIndexStats stats() const override;

  private:
    mutable std::shared_mutex mutex_;
    std::vector<backend::ChunkKey> keys_;
    std::vector<float> vectors_;
    std::size_t dimension_ = 0;
};

} // namespace rag::dense
