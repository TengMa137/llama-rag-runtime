#pragma once

#include <memory>

#include "rag/backend/candidate_backend.hpp"
#include "rag/ingestion/coordinator.hpp"

namespace rag::ingestion {

class Runtime {
  public:
    virtual ~Runtime() = default;
    [[nodiscard]] virtual Result<Submission> submit(IngestionInput input, bool asynchronous) = 0;
    [[nodiscard]] virtual Result<IngestionJob> erase(backend::DocumentKey document) = 0;
    [[nodiscard]] virtual Result<JobInfo> job(const JobId& id) const = 0;
    [[nodiscard]] virtual Result<IngestionJob> wait(const JobId& id) = 0;
    [[nodiscard]] virtual Result<std::vector<SearchResult>>
    search(backend::SearchRequest request) const = 0;
    [[nodiscard]] virtual Result<void> checkpoint() = 0;
};

} // namespace rag::ingestion
