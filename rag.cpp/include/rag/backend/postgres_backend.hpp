#pragma once

#include <memory>

#include "rag/backend/candidate_backend.hpp"
#include "rag/backend/postgres_config.hpp"

namespace rag::backend {

// PostgreSQL/pgvector CandidateBackend. The public header is libpq-free so the
// optional client ABI never leaks into portable callers or Android builds.
class PostgresBackend final : public CandidateBackend {
  public:
    ~PostgresBackend() override;
    PostgresBackend(const PostgresBackend&) = delete;
    PostgresBackend& operator=(const PostgresBackend&) = delete;

    [[nodiscard]] static Result<std::shared_ptr<PostgresBackend>> open(PostgresConfig config);

    Result<void> activate(PreparedDocument document, DocumentRevision revision) override;
    Result<bool> erase(DocumentKey document, DocumentRevision revision) override;
    [[nodiscard]] Result<CandidateList>
    lexical_candidates(const LexicalRequest& request) const override;
    [[nodiscard]] Result<CandidateList>
    dense_candidates(const DenseRequest& request) const override;
    [[nodiscard]] Result<CandidateBatch>
    hybrid_candidates(const HybridRequest& request) const override;
    [[nodiscard]] Result<std::vector<StoredChunk>> fetch(std::span<const ChunkKey> chunks,
                                                         FetchOptions options) const override;
    [[nodiscard]] Result<BackendStats> stats() const override;

  private:
    struct Impl;
    explicit PostgresBackend(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace rag::backend
