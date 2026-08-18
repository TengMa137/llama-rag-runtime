#pragma once

#include <memory>
#include <optional>
#include <string>

#include "rag/backend/embedded_backend.hpp"
#include "rag/ingestion/coordinator.hpp"
#include "rag/ingestion/embedded_durability.hpp"
#include "rag/ingestion/runtime.hpp"
#include "rag/retrieval/runtime.hpp"

namespace rag::ingestion {

struct EmbeddedRuntimeConfig {
    std::string checkpoint_path;
    std::string job_path;
    store::SyncMode sync_mode = store::SyncMode::flush;
    preparation::PrepareOptions preparation;
    std::optional<dense::AnyEmbedder> embedder;
    CoordinatorConfig coordinator;
    backend::EmbeddedMaintenancePolicy maintenance;
    JobRetentionPolicy job_retention;
    std::uint64_t checkpoint_wal_bytes = 256ULL * 1024ULL * 1024ULL;
};

// Composite ownership boundary for the embedded deployment. Construction
// restores the checkpoint first and then replays the surviving job-log tail.
class EmbeddedRuntime final : public Runtime {
  public:
    ~EmbeddedRuntime();
    EmbeddedRuntime(const EmbeddedRuntime&) = delete;
    EmbeddedRuntime& operator=(const EmbeddedRuntime&) = delete;

    [[nodiscard]] static Result<std::unique_ptr<EmbeddedRuntime>>
    open(EmbeddedRuntimeConfig config);

    [[nodiscard]] Result<Submission> submit(IngestionInput input, bool asynchronous) override;
    [[nodiscard]] Result<IngestionJob> erase(backend::DocumentKey document) override;
    [[nodiscard]] Result<JobInfo> job(const JobId& id) const override;
    [[nodiscard]] Result<IngestionJob> wait(const JobId& id) override;
    [[nodiscard]] Result<std::vector<SearchResult>>
    search(backend::SearchRequest request) const override;
    [[nodiscard]] Result<void> checkpoint() override;

    [[nodiscard]] backend::EmbeddedBackend& backend() noexcept { return *backend_; }
    [[nodiscard]] const backend::EmbeddedBackend& backend() const noexcept { return *backend_; }

  private:
    EmbeddedRuntime(EmbeddedRuntimeConfig config, std::shared_ptr<backend::EmbeddedBackend> backend,
                    std::shared_ptr<AppendOnlyJobStore> jobs,
                    std::unique_ptr<IngestionCoordinator> coordinator);
    [[nodiscard]] Result<void> checkpoint_if_needed();

    EmbeddedRuntimeConfig config_;
    std::shared_ptr<backend::EmbeddedBackend> backend_;
    std::shared_ptr<AppendOnlyJobStore> jobs_;
    std::unique_ptr<IngestionCoordinator> coordinator_;
};

} // namespace rag::ingestion
