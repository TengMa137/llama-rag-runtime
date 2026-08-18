#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "rag/backend/candidate_backend.hpp"

namespace rag::ingestion {

using JobId = std::string;

enum class JobOperation { upsert, erase };

enum class JobStatus {
    queued,
    chunking,
    embedding,
    publishing,
    ready,
    failed,
    superseded,
    cancelled,
};

[[nodiscard]] constexpr bool terminal(JobStatus status) noexcept {
    return status == JobStatus::ready || status == JobStatus::failed ||
           status == JobStatus::superseded || status == JobStatus::cancelled;
}

[[nodiscard]] std::string_view name(JobStatus status) noexcept;

struct IngestionInput {
    backend::DocumentKey document;
    std::string title;
    std::string content;
    Metadata metadata;
};

struct JobError {
    Errc code = Errc::ok;
    std::string message;
};

struct IngestionJob {
    JobId id;
    JobOperation operation = JobOperation::upsert;
    IngestionInput input;
    backend::DocumentRevision revision = 0;
    std::string content_hash;
    JobStatus status = JobStatus::queued;
    std::int64_t created_at_ms = 0;
    std::int64_t updated_at_ms = 0;
    std::optional<JobError> error;
    std::optional<backend::PreparedDocument> prepared;
    std::optional<bool> mutation_applied;
};

// Safe status projection for HTTP/C bindings. It intentionally excludes input
// content, prepared chunks, vectors, and provider configuration.
struct JobInfo {
    JobId id;
    backend::DocumentKey document;
    backend::DocumentRevision revision = 0;
    JobOperation operation = JobOperation::upsert;
    JobStatus status = JobStatus::queued;
    std::int64_t created_at_ms = 0;
    std::int64_t updated_at_ms = 0;
    std::optional<JobError> error;
};

[[nodiscard]] JobInfo info(const IngestionJob& job);

} // namespace rag::ingestion
