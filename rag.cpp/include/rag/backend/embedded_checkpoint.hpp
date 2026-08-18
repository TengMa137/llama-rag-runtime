#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "rag/backend/candidate_backend.hpp"

namespace rag::backend {

struct CheckpointDocument {
    std::shared_ptr<const PreparedDocument> prepared;
    DocumentRevision revision = 0;
};

// Portable durable state. Optimized dense indexes are intentionally excluded:
// they are disposable caches rebuilt from these normalized vectors.
struct EmbeddedCheckpoint {
    std::vector<CheckpointDocument> documents;
    std::map<DocumentKey, DocumentRevision> revisions;
    std::uint64_t generation = 0;
    std::uint64_t wal_position = 0;
};

class EmbeddedCheckpointStore {
  public:
    [[nodiscard]] static Result<void> save(const std::string& path,
                                           const EmbeddedCheckpoint& checkpoint);
    [[nodiscard]] static Result<EmbeddedCheckpoint> load(const std::string& path);
};

} // namespace rag::backend
