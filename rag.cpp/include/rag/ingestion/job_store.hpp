#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/ingestion/job.hpp"
#include "rag/store/wal.hpp"

namespace rag::ingestion {

struct JobRetentionPolicy {
    std::size_t max_terminal_jobs = 128;
    std::size_t max_terminal_bytes = 16 * 1024 * 1024;
};

class IngestionJobStore {
  public:
    virtual ~IngestionJobStore() = default;
    virtual Result<void> persist(const IngestionJob& job) = 0;
    [[nodiscard]] virtual Result<std::vector<IngestionJob>> load_latest() const = 0;
};

// Shared durable-record codec used by embedded WAL and PostgreSQL job stores.
// The JSON record is private persistence data, not an HTTP representation.
[[nodiscard]] Result<std::string> serialize_ingestion_job(const IngestionJob& job);
[[nodiscard]] Result<IngestionJob> deserialize_ingestion_job(std::string_view payload);

// Administrative read path used by migration and inspection. It accepts a
// torn final frame the same way recovery does, but never opens, truncates, or
// otherwise repairs the source log.
[[nodiscard]] Result<std::vector<IngestionJob>>
load_ingestion_jobs_read_only(const std::string& path);

// CRC-framed append-only job log. Every frame is a complete job snapshot, so
// replay needs no side files and a torn final append is safely ignored.
class AppendOnlyJobStore final : public IngestionJobStore {
  public:
    AppendOnlyJobStore() = default;
    ~AppendOnlyJobStore() override;
    AppendOnlyJobStore(const AppendOnlyJobStore&) = delete;
    AppendOnlyJobStore& operator=(const AppendOnlyJobStore&) = delete;

    [[nodiscard]] static Result<std::shared_ptr<AppendOnlyJobStore>>
    open(std::string path, store::SyncMode mode = store::SyncMode::flush);

    Result<void> persist(const IngestionJob& job) override;
    [[nodiscard]] Result<std::vector<IngestionJob>> load_latest() const override;
    [[nodiscard]] std::uint64_t size_bytes() const noexcept;
    // Atomically compact the represented prefix to all nonterminal jobs plus a
    // bounded recent terminal-status history, then retain every newer frame.
    // The position must be an exact frame boundary captured before publishing
    // the checkpoint that represents that prefix.
    [[nodiscard]] Result<void> truncate_prefix(std::uint64_t represented_position,
                                               JobRetentionPolicy retention = {});

  private:
    Result<void> open_file(std::string path, store::SyncMode mode);
    Result<void> sync_now() noexcept;

    mutable std::mutex mutex_;
    int fd_ = -1;
    std::string path_;
    store::SyncMode mode_ = store::SyncMode::flush;
    std::uint64_t bytes_ = 0;
};

} // namespace rag::ingestion
