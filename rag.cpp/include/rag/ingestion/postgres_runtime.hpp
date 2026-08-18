#pragma once

#include <memory>
#include <optional>

#include "rag/backend/postgres_backend.hpp"
#include "rag/dense/embedder.hpp"
#include "rag/ingestion/postgres_job_store.hpp"
#include "rag/ingestion/runtime.hpp"
#include "rag/preparation/document_preparer.hpp"

namespace rag::ingestion {

struct PostgresRuntimeConfig {
    backend::PostgresConfig database;
    preparation::PrepareOptions preparation;
    std::optional<dense::AnyEmbedder> embedder;
    CoordinatorConfig coordinator;
};

class PostgresRuntime final : public Runtime {
  public:
    ~PostgresRuntime() override;
    PostgresRuntime(const PostgresRuntime&) = delete;
    PostgresRuntime& operator=(const PostgresRuntime&) = delete;

    [[nodiscard]] static Result<std::unique_ptr<PostgresRuntime>>
    open(PostgresRuntimeConfig config);

    [[nodiscard]] Result<Submission> submit(IngestionInput input, bool asynchronous) override;
    [[nodiscard]] Result<IngestionJob> erase(backend::DocumentKey document) override;
    [[nodiscard]] Result<JobInfo> job(const JobId& id) const override;
    [[nodiscard]] Result<IngestionJob> wait(const JobId& id) override;
    [[nodiscard]] Result<std::vector<SearchResult>>
    search(backend::SearchRequest request) const override;
    [[nodiscard]] Result<void> checkpoint() override;

    [[nodiscard]] backend::PostgresBackend& backend() noexcept { return *backend_; }

  private:
    PostgresRuntime(PostgresRuntimeConfig config, std::shared_ptr<backend::PostgresBackend> backend,
                    std::shared_ptr<PostgresJobStore> jobs,
                    std::unique_ptr<IngestionCoordinator> coordinator);

    PostgresRuntimeConfig config_;
    std::shared_ptr<backend::PostgresBackend> backend_;
    std::shared_ptr<PostgresJobStore> jobs_;
    std::unique_ptr<IngestionCoordinator> coordinator_;
};

} // namespace rag::ingestion
