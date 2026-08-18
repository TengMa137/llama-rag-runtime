#pragma once

#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <vector>

#include "rag/dense/policy.hpp"

namespace rag::dense {

// Desktop-only optional adapter. No FAISS type crosses this header, keeping
// portable callers and Android independent of FAISS headers and ABI details.
class FaissIndex final : public DenseIndex {
  public:
    FaissIndex(DenseAlgorithm algorithm, DensePolicy::FaissParameters parameters);
    ~FaissIndex() override;
    FaissIndex(const FaissIndex&) = delete;
    FaissIndex& operator=(const FaissIndex&) = delete;

    [[nodiscard]] static Result<std::shared_ptr<FaissIndex>>
    from_serialized(std::string_view blob, std::span<const backend::ChunkKey> keys,
                    DenseAlgorithm algorithm, DensePolicy::FaissParameters parameters);

    Result<void> build(VectorSource source) override;
    [[nodiscard]] Result<backend::CandidateList> search(Vector query, AllowedIds allowed,
                                                        std::size_t k) const override;
    [[nodiscard]] DenseIndexStats stats() const override;
    [[nodiscard]] Result<std::string> serialize() const;
    [[nodiscard]] const std::vector<backend::ChunkKey>& keys() const noexcept { return keys_; }

  private:
    struct Impl;
    DenseAlgorithm algorithm_;
    DensePolicy::FaissParameters parameters_;
    mutable std::shared_mutex mutex_;
    std::unique_ptr<Impl> impl_;
    std::vector<backend::ChunkKey> keys_;
    std::size_t dimension_ = 0;
};

} // namespace rag::dense
