#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "rag/dense/native_hnsw.hpp"

namespace rag::dense {

[[nodiscard]] std::string hnsw_sidecar_path(std::string_view checkpoint_path);
[[nodiscard]] std::string hnsw_sidecar_fingerprint(VectorSource source,
                                                   std::string_view embedding_identity,
                                                   const NativeHnswPolicy& policy);
[[nodiscard]] Result<void> write_hnsw_sidecar(const std::string& path, std::string_view fingerprint,
                                              const NativeHnswIndex& index);
[[nodiscard]] Result<std::shared_ptr<NativeHnswIndex>>
load_hnsw_sidecar(const std::string& path, std::string_view expected_fingerprint,
                  std::span<const backend::ChunkKey> expected_keys, const NativeHnswPolicy& policy);

} // namespace rag::dense
