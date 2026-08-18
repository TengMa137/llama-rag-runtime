#include "rag/ingestion/job.hpp"

namespace rag::ingestion {

std::string_view name(JobStatus status) noexcept {
    switch (status) {
        case JobStatus::queued:
            return "queued";
        case JobStatus::chunking:
            return "chunking";
        case JobStatus::embedding:
            return "embedding";
        case JobStatus::publishing:
            return "publishing";
        case JobStatus::ready:
            return "ready";
        case JobStatus::failed:
            return "failed";
        case JobStatus::superseded:
            return "superseded";
        case JobStatus::cancelled:
            return "cancelled";
    }
    return "failed";
}

JobInfo info(const IngestionJob& job) {
    return {job.id,     job.input.document, job.revision,      job.operation,
            job.status, job.created_at_ms,  job.updated_at_ms, job.error};
}

} // namespace rag::ingestion
