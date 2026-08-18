#pragma once

#include <memory>

#include "rag/backend/postgres_config.hpp"
#include "rag/ingestion/job_store.hpp"

namespace rag::ingestion {

class PostgresJobStore final : public IngestionJobStore {
  public:
    ~PostgresJobStore() override;
    PostgresJobStore(const PostgresJobStore&) = delete;
    PostgresJobStore& operator=(const PostgresJobStore&) = delete;

    [[nodiscard]] static Result<std::shared_ptr<PostgresJobStore>>
    open(backend::PostgresConfig config);

    Result<void> persist(const IngestionJob& job) override;
    [[nodiscard]] Result<std::vector<IngestionJob>> load_latest() const override;

  private:
    struct Impl;
    explicit PostgresJobStore(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace rag::ingestion
