#pragma once

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

#include "rag/backend/candidate_backend.hpp"
#include "rag/dense/tiered_index.hpp"

namespace rag::backend {

struct ActiveDocumentState {
    DocumentRevision revision = 0;
    std::string content_hash;
};

struct EmbeddedMaintenancePolicy {
    std::size_t delta_chunk_limit = 10'000;
    double delta_base_fraction = 0.10;
    double tombstone_base_fraction = 0.10;
    std::size_t compaction_memory_budget = 1ULL << 30U;
    bool automatic_compaction = true;
    dense::DensePolicy dense;
};

// Data contract:
//   * the catalog and BM25 contain exactly one active revision per document;
//   * the immutable dense base may contain old rows hidden by tombstones;
//   * the exact dense delta contains every live row not represented by base;
//   * readers hold one immutable Generation and therefore never mix revisions.
class EmbeddedBackend final : public CandidateBackend {
  public:
    explicit EmbeddedBackend(EmbeddedMaintenancePolicy policy = {});
    ~EmbeddedBackend() override;

    [[nodiscard]] static Result<std::unique_ptr<EmbeddedBackend>>
    open_checkpoint(const std::string& path, EmbeddedMaintenancePolicy policy = {});

    Result<void> activate(PreparedDocument document, DocumentRevision revision) override;
    Result<bool> erase(DocumentKey document, DocumentRevision revision) override;

    [[nodiscard]] Result<CandidateList>
    lexical_candidates(const LexicalRequest& request) const override;
    [[nodiscard]] Result<CandidateList>
    dense_candidates(const DenseRequest& request) const override;
    [[nodiscard]] Result<std::vector<StoredChunk>> fetch(std::span<const ChunkKey> chunks,
                                                         FetchOptions options) const override;
    [[nodiscard]] Result<BackendStats> stats() const override;
    [[nodiscard]] Result<std::optional<ActiveDocumentState>>
    document_state(const DocumentKey& document) const;

    // Builds a new exact immutable base from a consistent live snapshot and
    // publishes it only if no writer changed that snapshot during the build.
    [[nodiscard]] Result<void> compact(bool override_memory_budget = false);
    [[nodiscard]] Result<void> checkpoint(const std::string& path, std::uint64_t wal_position);
    [[nodiscard]] std::uint64_t checkpoint_wal_position() const noexcept {
        return checkpoint_wal_position_.load();
    }

  private:
    struct ActiveDocument {
        DocumentRevision revision = 0;
        std::shared_ptr<const PreparedDocument> prepared;
    };
    struct Generation;

    [[nodiscard]] static Result<std::shared_ptr<const Generation>>
    build_generation(const std::map<DocumentKey, ActiveDocument>& documents,
                     std::shared_ptr<const dense::TieredDenseIndex> dense,
                     const std::unordered_set<DocumentKey>& delta_documents,
                     const std::unordered_set<ChunkKey>& tombstones, std::uint64_t generation);
    [[nodiscard]] static Result<void> validate(const PreparedDocument& document,
                                               DocumentRevision revision);
    void schedule_maintenance();
    void maintenance_loop();
    [[nodiscard]] Result<void> restore_checkpoint(const struct EmbeddedCheckpoint& checkpoint,
                                                  const std::string& checkpoint_path);

    mutable std::mutex writer_mutex_;
    EmbeddedMaintenancePolicy policy_;
    std::map<DocumentKey, DocumentRevision> latest_revisions_;
    std::shared_ptr<const Generation> active_;

    std::mutex maintenance_mutex_;
    std::condition_variable maintenance_ready_;
    bool maintenance_pending_ = false;
    bool stopping_ = false;
    std::thread maintenance_worker_;
    std::atomic<std::uint64_t> checkpoint_wal_position_{0};
};

} // namespace rag::backend
#include <atomic>
