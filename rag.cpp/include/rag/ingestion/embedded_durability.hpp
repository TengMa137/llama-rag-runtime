#pragma once

#include <string>

#include "rag/backend/embedded_backend.hpp"
#include "rag/ingestion/job_store.hpp"

namespace rag::ingestion {

// Publishes a checkpoint before discarding exactly the job-log prefix it
// represents. A failure after checkpoint publication leaves extra replayable
// records, never missing records.
[[nodiscard]] Result<void> checkpoint_embedded(backend::EmbeddedBackend& backend,
                                               AppendOnlyJobStore& jobs,
                                               const std::string& checkpoint_path,
                                               JobRetentionPolicy retention = {});

} // namespace rag::ingestion
