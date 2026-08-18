#include "rag/ingestion/postgres_runtime.hpp"

#include "rag/dense/simd.hpp"
#include "rag/retrieval/runtime.hpp"

namespace rag::ingestion {

PostgresRuntime::PostgresRuntime(PostgresRuntimeConfig config,
                                 std::shared_ptr<backend::PostgresBackend> backend,
                                 std::shared_ptr<PostgresJobStore> jobs,
                                 std::unique_ptr<IngestionCoordinator> coordinator)
    : config_(std::move(config)), backend_(std::move(backend)), jobs_(std::move(jobs)),
      coordinator_(std::move(coordinator)) {}

PostgresRuntime::~PostgresRuntime() = default;

Result<std::unique_ptr<PostgresRuntime>> PostgresRuntime::open(PostgresRuntimeConfig config) {
    auto backend = backend::PostgresBackend::open(config.database);
    if (!backend)
        return unexpected(backend.error());
    auto jobs = PostgresJobStore::open(config.database);
    if (!jobs)
        return unexpected(jobs.error());
    auto coordinator = IngestionCoordinator::open(*backend, *jobs, config.preparation,
                                                  config.embedder, config.coordinator);
    if (!coordinator)
        return unexpected(coordinator.error());
    return std::unique_ptr<PostgresRuntime>(new PostgresRuntime(
        std::move(config), std::move(*backend), std::move(*jobs), std::move(*coordinator)));
}

Result<Submission> PostgresRuntime::submit(IngestionInput input, bool asynchronous) {
    return coordinator_->submit(std::move(input), asynchronous);
}

Result<IngestionJob> PostgresRuntime::erase(backend::DocumentKey document) {
    return coordinator_->erase(std::move(document));
}

Result<JobInfo> PostgresRuntime::job(const JobId& id) const {
    auto found = coordinator_->get(id);
    if (!found)
        return unexpected(found.error());
    return info(*found);
}

Result<IngestionJob> PostgresRuntime::wait(const JobId& id) { return coordinator_->wait(id); }

Result<std::vector<SearchResult>> PostgresRuntime::search(backend::SearchRequest request) const {
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

Result<void> PostgresRuntime::checkpoint() {
    // Each activation and job transition is already committed transactionally.
    return {};
}

} // namespace rag::ingestion
