#pragma once

#include <memory>
#include <span>
#include <unordered_set>

#include "rag/dense/index.hpp"
#include "rag/dense/policy.hpp"

namespace rag::dense {

struct TieredIndexStats {
    std::size_t base_vectors = 0;
    std::size_t delta_vectors = 0;
    std::size_t tombstones = 0;
    std::size_t dimension = 0;
    std::size_t resident_bytes = 0;
    std::size_t mapped_bytes = 0;
    bool base_exact = true;
    std::string base_implementation = "native";
    std::string base_algorithm = "exact";
};

// Immutable two-tier dense index contract. The caller owns document/catalog
// semantics and supplies already-filtered IDs; this class owns a replaceable
// base, an exact mutable delta, tombstones, and deterministic cross-tier merge.
class TieredDenseIndex {
  public:
    [[nodiscard]] static Result<std::shared_ptr<const TieredDenseIndex>> empty();

    [[nodiscard]] Result<std::shared_ptr<const TieredDenseIndex>>
    with_delta(VectorSource live_delta, std::unordered_set<backend::ChunkKey> tombstones,
               std::size_t dimension) const;

    [[nodiscard]] Result<std::shared_ptr<const TieredDenseIndex>>
    compact(VectorSource live_vectors, std::size_t dimension, const DensePolicy& policy = {}) const;

    [[nodiscard]] static Result<std::shared_ptr<const TieredDenseIndex>>
    from_base(std::shared_ptr<DenseIndex> index, std::span<const backend::ChunkKey> keys,
              std::size_t dimension);

    [[nodiscard]] Result<backend::CandidateList>
    search(VectorView query, std::span<const backend::ChunkKey> base_allowed,
           std::span<const backend::ChunkKey> delta_allowed, std::size_t k) const;

    [[nodiscard]] bool contains_base(const backend::ChunkKey& key) const;
    [[nodiscard]] bool contains_delta(const backend::ChunkKey& key) const;
    [[nodiscard]] const std::unordered_set<backend::ChunkKey>& tombstones() const noexcept {
        return tombstones_;
    }
    [[nodiscard]] std::shared_ptr<const DenseIndex> base_index() const noexcept;
    [[nodiscard]] TieredIndexStats stats() const;

  private:
    struct Base;
    std::shared_ptr<const Base> base_;
    std::shared_ptr<NativeExactIndex> delta_;
    std::unordered_set<backend::ChunkKey> delta_keys_;
    std::unordered_set<backend::ChunkKey> tombstones_;
    std::size_t dimension_ = 0;
};

using TieredExactIndex [[deprecated("use TieredDenseIndex")]] = TieredDenseIndex;

} // namespace rag::dense
