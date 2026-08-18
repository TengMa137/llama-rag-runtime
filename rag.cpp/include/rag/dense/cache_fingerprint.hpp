#pragma once

#include <string>
#include <string_view>

#include "rag/dense/index.hpp"

namespace rag::dense {

struct DenseCacheIdentity {
    std::string_view implementation;
    std::string_view algorithm;
    std::uint32_t version = 0;
    std::string parameters;
};

// Implementation-cache identity, not a durable corpus hash. It deliberately
// includes ordered public keys and raw normalized f32 bytes so stale caches
// fail closed after any active-generation change.
[[nodiscard]] std::string dense_cache_fingerprint(VectorSource source,
                                                  std::string_view embedding_identity,
                                                  const DenseCacheIdentity& identity);

} // namespace rag::dense
