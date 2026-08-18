#include "rag/ingestion/embedded_runtime.hpp"

#include <filesystem>
#include <mutex>

#include "rag/dense/simd.hpp"

namespace rag::ingestion {

EmbeddedRuntime::EmbeddedRuntime(EmbeddedRuntimeConfig config,
                                 std::shared_ptr<backend::EmbeddedBackend> backend,
                                 std::shared_ptr<AppendOnlyJobStore> jobs,
                                 std::unique_ptr<IngestionCoordinator> coordinator)
    : config_(std::move(config)), backend_(std::move(backend)), jobs_(std::move(jobs)),
      coordinator_(std::move(coordinator)) {}

EmbeddedRuntime::~EmbeddedRuntime() = default;

Result<std::unique_ptr<EmbeddedRuntime>> EmbeddedRuntime::open(EmbeddedRuntimeConfig config) {
    if (config.checkpoint_path.empty() || config.checkpoint_wal_bytes == 0)
        return fail<std::unique_ptr<EmbeddedRuntime>>(
            Errc::invalid_argument, "checkpoint path and non-zero WAL limit are required");
    if (config.job_path.empty())
        config.job_path = config.checkpoint_path + ".jobs";
    try {
        for (const auto& path : {config.checkpoint_path, config.job_path}) {
            const std::filesystem::path value(path);
            if (value.has_parent_path())
                std::filesystem::create_directories(value.parent_path());
        }
    } catch (const std::exception& error) {
        return fail<std::unique_ptr<EmbeddedRuntime>>(
            Errc::io_error,
            "embedded runtime directory creation failed: " + std::string(error.what()));
    }

    std::shared_ptr<backend::EmbeddedBackend> backend;
    if (std::filesystem::exists(config.checkpoint_path)) {
        auto reopened =
            backend::EmbeddedBackend::open_checkpoint(config.checkpoint_path, config.maintenance);
        if (!reopened)
            return unexpected(reopened.error());
        backend = std::shared_ptr<backend::EmbeddedBackend>(std::move(*reopened));
    } else {
        backend = std::make_shared<backend::EmbeddedBackend>(config.maintenance);
    }
    auto jobs = AppendOnlyJobStore::open(config.job_path, config.sync_mode);
    if (!jobs)
        return unexpected(jobs.error());
    const auto maintenance_lock = std::make_shared<std::mutex>();
    const auto prior_ready = config.coordinator.on_ready;
    config.coordinator.on_ready = [backend, jobs = *jobs, path = config.checkpoint_path,
                                   threshold = config.checkpoint_wal_bytes,
                                   retention = config.job_retention, maintenance_lock,
                                   prior_ready] {
        if (prior_ready)
            prior_ready();
        if (jobs->size_bytes() < threshold)
            return;
        std::lock_guard lock(*maintenance_lock);
        if (jobs->size_bytes() >= threshold)
            (void)checkpoint_embedded(*backend, *jobs, path, retention);
    };
    auto coordinator = IngestionCoordinator::open(backend, *jobs, config.preparation,
                                                  config.embedder, config.coordinator);
    if (!coordinator)
        return unexpected(coordinator.error());
    return std::unique_ptr<EmbeddedRuntime>(new EmbeddedRuntime(
        std::move(config), std::move(backend), std::move(*jobs), std::move(*coordinator)));
}

Result<Submission> EmbeddedRuntime::submit(IngestionInput input, bool asynchronous) {
    auto submitted = coordinator_->submit(std::move(input), asynchronous);
    if (!submitted)
        return unexpected(submitted.error());
    if (!asynchronous && terminal(submitted->job.status))
        if (auto maintained = checkpoint_if_needed(); !maintained)
            return unexpected(maintained.error());
    return submitted;
}

Result<IngestionJob> EmbeddedRuntime::erase(backend::DocumentKey document) {
    auto erased = coordinator_->erase(std::move(document));
    if (!erased)
        return unexpected(erased.error());
    if (auto maintained = checkpoint_if_needed(); !maintained)
        return unexpected(maintained.error());
    return erased;
}

Result<JobInfo> EmbeddedRuntime::job(const JobId& id) const {
    auto found = coordinator_->get(id);
    if (!found)
        return unexpected(found.error());
    return info(*found);
}

Result<IngestionJob> EmbeddedRuntime::wait(const JobId& id) {
    auto completed = coordinator_->wait(id);
    if (!completed)
        return unexpected(completed.error());
    if (terminal(completed->status))
        if (auto maintained = checkpoint_if_needed(); !maintained)
            return unexpected(maintained.error());
    return completed;
}

Result<std::vector<SearchResult>> EmbeddedRuntime::search(backend::SearchRequest request) const {
    if (request.mode != backend::SearchMode::lexical && !request.embedding) {
        if (!config_.embedder)
            return fail<std::vector<SearchResult>>(Errc::unavailable,
                                                   "dense search requires an embedding provider");
        auto embedded = config_.embedder->embed_one(request.query);
        if (!embedded)
            return unexpected(embedded.error());
        dense::normalize(*embedded);
        request.embedding = std::move(*embedded);
    }
    return retrieval::search(*backend_, request);
}

Result<void> EmbeddedRuntime::checkpoint() {
    return checkpoint_embedded(*backend_, *jobs_, config_.checkpoint_path, config_.job_retention);
}

Result<void> EmbeddedRuntime::checkpoint_if_needed() {
    if (jobs_->size_bytes() < config_.checkpoint_wal_bytes)
        return {};
    return checkpoint();
}

} // namespace rag::ingestion
