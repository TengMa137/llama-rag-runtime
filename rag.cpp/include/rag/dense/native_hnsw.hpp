#pragma once

#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <vector>

#include "rag/dense/index.hpp"
#include "rag/index/hnsw.hpp"

namespace rag::dense {

struct NativeHnswPolicy {
    std::size_t neighbors = 16;
    std::size_t ef_construction = 200;
    std::size_t ef_search = 64;
    std::uint64_t seed = 0x9E3779B97F4A7C15ULL;
};

// Immutable-after-build native HNSW adapter. The graph owns one f32 matrix and
// no SQ8/PQ mirror; the exact scan of that same matrix is the selective-filter
// completeness fallback.
class NativeHnswIndex final : public DenseIndex {
  public:
    explicit NativeHnswIndex(NativeHnswPolicy policy = {});

    [[nodiscard]] static Result<std::shared_ptr<NativeHnswIndex>>
    from_serialized(std::string_view blob, std::span<const backend::ChunkKey> keys,
                    NativeHnswPolicy policy);

    Result<void> build(VectorSource source) override;
    [[nodiscard]] Result<backend::CandidateList> search(Vector query, AllowedIds allowed,
                                                        std::size_t k) const override;
    [[nodiscard]] DenseIndexStats stats() const override;
    [[nodiscard]] Result<std::string> serialize() const;
    [[nodiscard]] const std::vector<backend::ChunkKey>& keys() const noexcept { return keys_; }

  private:
    [[nodiscard]] static Result<void> validate_policy(const NativeHnswPolicy& policy);

    NativeHnswPolicy policy_;
    mutable std::shared_mutex mutex_;
    std::unique_ptr<index::HnswIndex> graph_;
    std::vector<backend::ChunkKey> keys_;
    std::size_t dimension_ = 0;
};

} // namespace rag::dense
