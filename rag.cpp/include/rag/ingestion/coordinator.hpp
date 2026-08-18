#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rag/backend/candidate_backend.hpp"
#include "rag/dense/embedder.hpp"
#include "rag/ingestion/job_store.hpp"
#include "rag/preparation/document_preparer.hpp"

namespace rag::ingestion {

struct CoordinatorConfig {
    std::size_t worker_count = 1;
    std::size_t queue_capacity = 64;
    std::function<void()> on_ready;
};

struct Submission {
    IngestionJob job;
    bool unchanged = false;
    bool existing_pending = false;
};

class IngestionCoordinator {
  public:
    ~IngestionCoordinator();
    IngestionCoordinator(const IngestionCoordinator&) = delete;
    IngestionCoordinator& operator=(const IngestionCoordinator&) = delete;

    [[nodiscard]] static Result<std::unique_ptr<IngestionCoordinator>>
    open(std::shared_ptr<backend::CandidateBackend> backend,
         std::shared_ptr<IngestionJobStore> store, preparation::PrepareOptions preparation,
         std::optional<dense::AnyEmbedder> embedder = std::nullopt, CoordinatorConfig config = {});

    [[nodiscard]] Result<Submission> submit(IngestionInput input, bool asynchronous);
    [[nodiscard]] Result<IngestionJob> erase(backend::DocumentKey document);
    [[nodiscard]] Result<IngestionJob> get(const JobId& id) const;
    [[nodiscard]] Result<IngestionJob> wait(const JobId& id);
    void shutdown() noexcept;

  private:
    IngestionCoordinator(std::shared_ptr<backend::CandidateBackend> backend,
                         std::shared_ptr<IngestionJobStore> store,
                         preparation::PrepareOptions preparation,
                         std::optional<dense::AnyEmbedder> embedder, CoordinatorConfig config);

    Result<void> recover();
    void worker_loop();
    void execute(const JobId& id);
    Result<bool> transition(const JobId& id, JobStatus status,
                            std::optional<JobError> error = std::nullopt,
                            std::optional<backend::PreparedDocument> prepared = std::nullopt,
                            std::optional<bool> mutation_applied = std::nullopt);
    void fail_job(const JobId& id, const Error& error);
    void fail_volatile(const JobId& id, const Error& error);

    std::shared_ptr<backend::CandidateBackend> backend_;
    std::shared_ptr<IngestionJobStore> store_;
    preparation::PrepareOptions preparation_;
    std::optional<dense::AnyEmbedder> embedder_;
    CoordinatorConfig config_;

    mutable std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable state_changed_;
    bool stopping_ = false;
    std::deque<JobId> queue_;
    std::unordered_map<JobId, IngestionJob> jobs_;
    std::unordered_map<backend::DocumentKey, backend::DocumentRevision> latest_revision_;
    std::unordered_map<backend::DocumentKey, JobId> latest_job_;
    std::vector<std::thread> workers_;
};

} // namespace rag::ingestion
