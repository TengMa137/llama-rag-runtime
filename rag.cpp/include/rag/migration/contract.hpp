#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "rag/backend/candidate_backend.hpp"

namespace rag::migration {

struct RevisionedDocument {
    backend::PreparedDocument document;
    backend::DocumentRevision revision = 0;
};

struct DocumentBatch {
    std::vector<RevisionedDocument> documents;
    std::string next_cursor;
    bool complete = false;
};

// Canonical corpus identity and streaming verification values. Checksums cover
// active public IDs, revisions, document hashes, and normalized vector bytes.
struct CorpusAudit {
    std::size_t documents = 0;
    std::size_t chunks = 0;
    std::size_t dimension = 0;
    std::string embedding_identity;
    std::string chunking_fingerprint;
    std::string document_checksum;
    std::string vector_checksum;
    std::string fingerprint;

    friend bool operator==(const CorpusAudit&, const CorpusAudit&) = default;
};

struct MigrationProgress {
    std::string run_id;
    std::string source_fingerprint;
    std::string last_document;
    std::size_t documents = 0;
    std::size_t chunks = 0;
    bool complete = false;
};

struct MigrationOptions {
    std::size_t batch_size = 256;
    std::size_t sample_searches = 16;
};

struct MigrationReport {
    std::string id;
    std::string direction;
    std::string source;
    std::string destination;
    bool resumed = false;
    bool complete = false;
    std::size_t batches = 0;
    std::size_t sampled_searches = 0;
    CorpusAudit source_audit;
    CorpusAudit destination_audit;
};

// Administrative data plane. Implementations must return active documents in
// strictly increasing DocumentKey order and make write_batch idempotent for a
// repeated run/cursor pair. CandidateBackend is deliberately not stretched to
// include bulk export or migration progress.
class Endpoint {
  public:
    virtual ~Endpoint() = default;
    [[nodiscard]] virtual std::string description() const = 0;
    [[nodiscard]] virtual Result<DocumentBatch> read_batch(std::string_view after,
                                                           std::size_t limit) const = 0;
    [[nodiscard]] virtual Result<std::optional<MigrationProgress>>
    progress(std::string_view run_id) const = 0;
    [[nodiscard]] virtual Result<void> write_batch(std::string_view run_id,
                                                   const CorpusAudit& source,
                                                   const DocumentBatch& batch,
                                                   const MigrationProgress& progress) = 0;
    [[nodiscard]] virtual Result<void> finish(std::string_view run_id, const CorpusAudit& source,
                                              const MigrationProgress& progress) = 0;
    [[nodiscard]] virtual Result<backend::CandidateList> exact_candidates(VectorView query,
                                                                          std::size_t k) const = 0;
};

[[nodiscard]] Result<CorpusAudit> audit(const Endpoint& endpoint, std::size_t batch_size);
[[nodiscard]] Result<MigrationReport> migrate(Endpoint& source, Endpoint& destination,
                                              std::string direction, MigrationOptions options = {});
[[nodiscard]] std::string json_report(const MigrationReport& report);

} // namespace rag::migration
