#include "rag/ingestion/coordinator.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>

#include "rag/core/keys.hpp"

namespace rag::ingestion {
namespace {

std::int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

JobId next_job_id() {
    static std::atomic<std::uint64_t> sequence{0};
    const auto clock = std::chrono::steady_clock::now().time_since_epoch().count();
    return "job_" + std::to_string(clock) + "_" + std::to_string(sequence.fetch_add(1));
}

JobError bounded_error(const Error& error) {
    constexpr std::size_t kMaxErrorBytes = 1024;
    std::string message = error.message;
    if (message.size() > kMaxErrorBytes)
        message.resize(kMaxErrorBytes);
    return {error.code, std::move(message)};
}

} // namespace

IngestionCoordinator::IngestionCoordinator(std::shared_ptr<backend::CandidateBackend> backend,
                                           std::shared_ptr<IngestionJobStore> store,
                                           preparation::PrepareOptions preparation,
                                           std::optional<dense::AnyEmbedder> embedder,
                                           CoordinatorConfig config)
    : backend_(std::move(backend)), store_(std::move(store)), preparation_(std::move(preparation)),
      embedder_(std::move(embedder)), config_(config) {}

IngestionCoordinator::~IngestionCoordinator() { shutdown(); }

Result<std::unique_ptr<IngestionCoordinator>>
IngestionCoordinator::open(std::shared_ptr<backend::CandidateBackend> backend,
                           std::shared_ptr<IngestionJobStore> store,
                           preparation::PrepareOptions preparation,
                           std::optional<dense::AnyEmbedder> embedder, CoordinatorConfig config) {
    if (!backend || !store)
        return fail<std::unique_ptr<IngestionCoordinator>>(Errc::invalid_argument,
                                                           "backend and job store are required");
    if (config.worker_count == 0 || config.worker_count > 4 || config.queue_capacity == 0)
        return fail<std::unique_ptr<IngestionCoordinator>>(
            Errc::invalid_argument, "worker count must be 1-4 and queue capacity must be non-zero");
    auto coordinator = std::unique_ptr<IngestionCoordinator>(new IngestionCoordinator(
        std::move(backend), std::move(store), std::move(preparation), std::move(embedder), config));
    if (auto recovered = coordinator->recover(); !recovered)
        return unexpected(recovered.error());
    coordinator->workers_.reserve(config.worker_count);
    for (std::size_t index = 0; index < config.worker_count; ++index)
        coordinator->workers_.emplace_back([self = coordinator.get()] { self->worker_loop(); });
    return coordinator;
}

Result<void> IngestionCoordinator::recover() {
    auto loaded = store_->load_latest();
    if (!loaded)
        return unexpected(loaded.error());
    for (auto& job : *loaded) {
        const auto found = latest_revision_.find(job.input.document);
        if (found == latest_revision_.end() || job.revision > found->second) {
            latest_revision_[job.input.document] = job.revision;
            latest_job_[job.input.document] = job.id;
        }
        jobs_[job.id] = std::move(job);
    }

    std::unordered_map<backend::DocumentKey, JobId> active_ready;
    for (const auto& [id, job] : jobs_) {
        if (job.status != JobStatus::ready)
            continue;
        const auto found = active_ready.find(job.input.document);
        if (found == active_ready.end() || jobs_.at(found->second).revision < job.revision)
            active_ready[job.input.document] = id;
    }
    std::vector<JobId> ready;
    ready.reserve(active_ready.size());
    for (const auto& [document, id] : active_ready)
        ready.push_back(id);
    std::sort(ready.begin(), ready.end(), [&](const auto& left, const auto& right) {
        const auto& a = jobs_.at(left);
        const auto& b = jobs_.at(right);
        if (a.revision != b.revision)
            return a.revision < b.revision;
        return a.id < b.id;
    });
    for (const auto& id : ready) {
        auto& job = jobs_.at(id);
        if (job.operation == JobOperation::erase) {
            if (auto erased = backend_->erase(job.input.document, job.revision); !erased)
                return unexpected(erased.error());
        } else {
            if (!job.prepared)
                return fail<void>(Errc::corrupt_index,
                                  "ready job has no prepared activation record");
            if (auto activated = backend_->activate(*job.prepared, job.revision); !activated)
                return unexpected(activated.error());
        }
    }

    std::vector<JobId> recovery_deletions;
    for (auto& [id, job] : jobs_) {
        if (terminal(job.status))
            continue;
        if (latest_revision_[job.input.document] != job.revision) {
            job.status = JobStatus::superseded;
            job.updated_at_ms = now_ms();
            if (auto saved = store_->persist(job); !saved)
                return saved;
            continue;
        }
        job.status = JobStatus::queued;
        job.updated_at_ms = now_ms();
        job.error.reset();
        job.prepared.reset();
        if (auto saved = store_->persist(job); !saved)
            return saved;
        if (job.operation == JobOperation::erase)
            recovery_deletions.push_back(id);
        else
            queue_.push_back(id);
    }
    for (const auto& id : recovery_deletions)
        execute(id);
    return {};
}

Result<Submission> IngestionCoordinator::submit(IngestionInput input, bool asynchronous) {
    if (input.document.empty() || input.content.empty())
        return fail<Submission>(Errc::invalid_argument, "document and content are required");
    input.content = normalize_source_text(input.content);
    const std::string fingerprint =
        document_content_hash(input.title, input.content, input.metadata);

    IngestionJob created;
    {
        std::unique_lock lock(mutex_);
        if (stopping_)
            return fail<Submission>(Errc::unavailable, "ingestion coordinator is stopping");
        const auto latest_id = latest_job_.find(input.document);
        if (latest_id != latest_job_.end()) {
            const auto& latest = jobs_.at(latest_id->second);
            if (latest.content_hash == fingerprint) {
                if (latest.status == JobStatus::ready)
                    return Submission{latest, true, false};
                if (!terminal(latest.status)) {
                    if (asynchronous)
                        return Submission{latest, false, true};
                    const JobId existing = latest.id;
                    lock.unlock();
                    auto completed = wait(existing);
                    if (!completed)
                        return unexpected(completed.error());
                    return Submission{std::move(*completed), false, true};
                }
            }
        }
        if (asynchronous && queue_.size() >= config_.queue_capacity)
            return fail<Submission>(Errc::unavailable, "ingestion queue is full");

        created.id = next_job_id();
        created.input = std::move(input);
        created.revision = latest_revision_[created.input.document] + 1;
        created.content_hash = fingerprint;
        created.status = JobStatus::queued;
        created.created_at_ms = created.updated_at_ms = now_ms();
        if (auto saved = store_->persist(created); !saved)
            return unexpected(saved.error());

        if (latest_id != latest_job_.end()) {
            auto& older = jobs_.at(latest_id->second);
            if (!terminal(older.status)) {
                auto superseded = older;
                superseded.status = JobStatus::superseded;
                superseded.updated_at_ms = now_ms();
                superseded.input.content.clear();
                superseded.prepared.reset();
                if (auto saved = store_->persist(superseded); !saved)
                    return unexpected(saved.error());
                older = std::move(superseded);
            }
        }
        latest_revision_[created.input.document] = created.revision;
        latest_job_[created.input.document] = created.id;
        jobs_[created.id] = created;
        if (asynchronous) {
            queue_.push_back(created.id);
            work_ready_.notify_one();
            state_changed_.notify_all();
            return Submission{created, false, false};
        }
    }

    execute(created.id);
    auto completed = get(created.id);
    if (!completed)
        return unexpected(completed.error());
    return Submission{std::move(*completed), false, false};
}

Result<IngestionJob> IngestionCoordinator::erase(backend::DocumentKey document) {
    if (document.empty())
        return fail<IngestionJob>(Errc::invalid_argument, "document is required");
    IngestionJob deletion;
    {
        std::lock_guard lock(mutex_);
        if (stopping_)
            return fail<IngestionJob>(Errc::unavailable, "ingestion coordinator is stopping");
        deletion.id = next_job_id();
        deletion.operation = JobOperation::erase;
        deletion.input.document = document;
        deletion.revision = latest_revision_[document] + 1;
        deletion.content_hash = "deleted";
        deletion.status = JobStatus::queued;
        deletion.created_at_ms = deletion.updated_at_ms = now_ms();
        if (auto saved = store_->persist(deletion); !saved)
            return unexpected(saved.error());

        const auto latest_id = latest_job_.find(document);
        if (latest_id != latest_job_.end()) {
            auto& older = jobs_.at(latest_id->second);
            if (!terminal(older.status)) {
                auto superseded = older;
                superseded.status = JobStatus::superseded;
                superseded.updated_at_ms = now_ms();
                superseded.input.content.clear();
                superseded.prepared.reset();
                if (auto saved = store_->persist(superseded); !saved)
                    return unexpected(saved.error());
                older = std::move(superseded);
            }
        }
        latest_revision_[document] = deletion.revision;
        latest_job_[document] = deletion.id;
        jobs_[deletion.id] = deletion;
    }
    execute(deletion.id);
    return get(deletion.id);
}

Result<IngestionJob> IngestionCoordinator::get(const JobId& id) const {
    std::lock_guard lock(mutex_);
    const auto found = jobs_.find(id);
    if (found == jobs_.end())
        return fail<IngestionJob>(Errc::not_found, "ingestion job was not found");
    return found->second;
}

Result<IngestionJob> IngestionCoordinator::wait(const JobId& id) {
    std::unique_lock lock(mutex_);
    const auto exists = jobs_.find(id);
    if (exists == jobs_.end())
        return fail<IngestionJob>(Errc::not_found, "ingestion job was not found");
    state_changed_.wait(lock, [&] {
        const auto found = jobs_.find(id);
        return found == jobs_.end() || terminal(found->second.status) || stopping_;
    });
    const auto found = jobs_.find(id);
    if (found == jobs_.end())
        return fail<IngestionJob>(Errc::not_found, "ingestion job was not found");
    return found->second;
}

Result<bool> IngestionCoordinator::transition(const JobId& id, JobStatus status,
                                              std::optional<JobError> error,
                                              std::optional<backend::PreparedDocument> prepared,
                                              std::optional<bool> mutation_applied) {
    std::lock_guard lock(mutex_);
    const auto found = jobs_.find(id);
    if (found == jobs_.end())
        return fail<bool>(Errc::not_found, "ingestion job was not found");
    auto next = found->second;
    if (terminal(next.status))
        return false;
    if (latest_revision_[next.input.document] != next.revision) {
        next.status = JobStatus::superseded;
        next.updated_at_ms = now_ms();
        next.error.reset();
        next.input.content.clear();
        next.prepared.reset();
        if (auto saved = store_->persist(next); !saved)
            return unexpected(saved.error());
        found->second = std::move(next);
        state_changed_.notify_all();
        return false;
    }
    next.status = status;
    next.updated_at_ms = now_ms();
    next.error = std::move(error);
    if (prepared)
        next.prepared = std::move(prepared);
    if (mutation_applied)
        next.mutation_applied = mutation_applied;
    if (terminal(status)) {
        next.input.content.clear();
        if (status != JobStatus::ready)
            next.prepared.reset();
    }
    if (auto saved = store_->persist(next); !saved)
        return unexpected(saved.error());
    found->second = std::move(next);
    state_changed_.notify_all();
    return true;
}

void IngestionCoordinator::fail_job(const JobId& id, const Error& error) {
    auto failed = transition(id, JobStatus::failed, bounded_error(error));
    if (!failed)
        fail_volatile(id, failed.error());
}

void IngestionCoordinator::fail_volatile(const JobId& id, const Error& error) {
    std::lock_guard lock(mutex_);
    const auto found = jobs_.find(id);
    if (found == jobs_.end() || terminal(found->second.status))
        return;
    found->second.status = latest_revision_[found->second.input.document] == found->second.revision
                               ? JobStatus::failed
                               : JobStatus::superseded;
    found->second.updated_at_ms = now_ms();
    found->second.error = bounded_error(error);
    found->second.input.content.clear();
    found->second.prepared.reset();
    state_changed_.notify_all();
}

void IngestionCoordinator::execute(const JobId& id) {
    JobOperation operation;
    {
        std::lock_guard lock(mutex_);
        const auto found = jobs_.find(id);
        if (found == jobs_.end() || terminal(found->second.status))
            return;
        operation = found->second.operation;
    }
    if (operation == JobOperation::erase) {
        auto publishing = transition(id, JobStatus::publishing);
        if (!publishing) {
            fail_volatile(id, publishing.error());
            return;
        }
        if (!*publishing)
            return;
        Error publication_error;
        bool publication_failed = false;
        bool mutation_applied = false;
        {
            std::lock_guard lock(mutex_);
            const auto& job = jobs_.at(id);
            auto erased = backend_->erase(job.input.document, job.revision);
            if (!erased) {
                publication_error = erased.error();
                publication_failed = true;
            } else
                mutation_applied = *erased;
        }
        if (publication_failed) {
            fail_job(id, publication_error);
            return;
        }
        auto ready = transition(id, JobStatus::ready, std::nullopt, std::nullopt, mutation_applied);
        if (!ready)
            fail_volatile(id, ready.error());
        else if (*ready && config_.on_ready)
            config_.on_ready();
        return;
    }

    auto chunking = transition(id, JobStatus::chunking);
    if (!chunking) {
        fail_volatile(id, chunking.error());
        return;
    }
    if (!*chunking)
        return;
    IngestionInput input;
    {
        std::lock_guard lock(mutex_);
        input = jobs_.at(id).input;
    }
    auto prepared = preparation::prepare_chunks(input.document, input.content, input.metadata,
                                                input.title, preparation_.chunking);
    if (!prepared) {
        fail_job(id, prepared.error());
        return;
    }
    auto embedding = transition(id, JobStatus::embedding);
    if (!embedding) {
        fail_volatile(id, embedding.error());
        return;
    }
    if (!*embedding)
        return;
    if (embedder_) {
        if (auto result = preparation::embed_document(*prepared, *embedder_,
                                                      preparation_.embedding_batch_size);
            !result) {
            fail_job(id, result.error());
            return;
        }
    }
    auto publishing = transition(id, JobStatus::publishing, std::nullopt, *prepared);
    if (!publishing) {
        fail_volatile(id, publishing.error());
        return;
    }
    if (!*publishing)
        return;

    Error publication_error;
    bool publication_failed = false;
    {
        std::lock_guard lock(mutex_);
        const auto found = jobs_.find(id);
        if (found == jobs_.end() || terminal(found->second.status) ||
            latest_revision_[found->second.input.document] != found->second.revision)
            return;
        auto activated = backend_->activate(*prepared, found->second.revision);
        if (!activated) {
            publication_error = activated.error();
            publication_failed = true;
        }
    }
    if (publication_failed) {
        fail_job(id, publication_error);
        return;
    }
    auto ready = transition(id, JobStatus::ready, std::nullopt, std::move(*prepared));
    if (!ready)
        fail_volatile(id, ready.error());
    else if (*ready && config_.on_ready)
        config_.on_ready();
}

void IngestionCoordinator::worker_loop() {
    while (true) {
        JobId id;
        {
            std::unique_lock lock(mutex_);
            work_ready_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
            if (stopping_)
                return;
            id = std::move(queue_.front());
            queue_.pop_front();
        }
        execute(id);
    }
}

void IngestionCoordinator::shutdown() noexcept {
    {
        std::lock_guard lock(mutex_);
        if (stopping_)
            return;
        stopping_ = true;
    }
    work_ready_.notify_all();
    state_changed_.notify_all();
    for (auto& worker : workers_)
        if (worker.joinable())
            worker.join();
    workers_.clear();
    std::lock_guard lock(mutex_);
    queue_.clear();
}

} // namespace rag::ingestion
