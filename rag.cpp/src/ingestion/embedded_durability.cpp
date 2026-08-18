#include "rag/ingestion/embedded_durability.hpp"

namespace rag::ingestion {

Result<void> checkpoint_embedded(backend::EmbeddedBackend& backend, AppendOnlyJobStore& jobs,
                                 const std::string& checkpoint_path, JobRetentionPolicy retention) {
    const std::uint64_t represented_position = jobs.size_bytes();
    if (auto saved = backend.checkpoint(checkpoint_path, represented_position); !saved)
        return saved;
    return jobs.truncate_prefix(represented_position, retention);
}

} // namespace rag::ingestion
