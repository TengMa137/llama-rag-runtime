#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "rag/dense/index.hpp"
#include "rag/store/container_view.hpp"

namespace rag::dense {

[[nodiscard]] std::string exact_sidecar_path(std::string_view checkpoint_path);
[[nodiscard]] std::string exact_sidecar_fingerprint(VectorSource source,
                                                    std::string_view embedding_identity);
[[nodiscard]] Result<void> write_exact_sidecar(const std::string& path,
                                               std::string_view fingerprint, VectorSource source);

// Read-only exact index over a validated, disposable mapped sidecar. Keys are
// owned because candidate DTOs outlive searches; the f32 matrix remains mapped.
class MappedExactIndex final : public DenseIndex {
  public:
    [[nodiscard]] static Result<std::shared_ptr<MappedExactIndex>>
    open(const std::string& path, std::string_view expected_fingerprint);

    Result<void> build(VectorSource source) override;
    [[nodiscard]] Result<backend::CandidateList> search(Vector query, AllowedIds allowed,
                                                        std::size_t k) const override;
    [[nodiscard]] DenseIndexStats stats() const override;
    [[nodiscard]] const std::vector<backend::ChunkKey>& keys() const noexcept { return keys_; }

  private:
    store::ContainerView mapping_;
    std::vector<backend::ChunkKey> keys_;
    std::string_view vectors_;
    std::size_t dimension_ = 0;
};

} // namespace rag::dense
