#pragma once

#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "rag/dense/faiss_index.hpp"

namespace rag::dense {

// FAISS sidecars are disposable implementation caches. The portable .ragdb
// remains authoritative and can always rebuild one from its normalized vectors.
[[nodiscard]] std::string faiss_sidecar_path(std::string_view checkpoint_path,
                                             DenseAlgorithm algorithm);
[[nodiscard]] std::string faiss_sidecar_fingerprint(VectorSource source,
                                                    std::string_view embedding_identity,
                                                    DenseAlgorithm algorithm,
                                                    const DensePolicy::FaissParameters& parameters);
[[nodiscard]] Result<void>
write_faiss_sidecar(const std::string& path, std::string_view fingerprint, const FaissIndex& index);
[[nodiscard]] Result<std::shared_ptr<FaissIndex>>
load_faiss_sidecar(const std::string& path, std::string_view expected_fingerprint,
                   std::span<const backend::ChunkKey> expected_keys, DenseAlgorithm algorithm,
                   const DensePolicy::FaissParameters& parameters);

} // namespace rag::dense
