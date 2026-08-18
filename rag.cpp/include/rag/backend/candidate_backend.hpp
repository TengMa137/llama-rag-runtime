#pragma once
// Backend-neutral records and candidate retrieval contract.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/document.hpp"
#include "rag/core/types.hpp"

namespace rag::backend {

using DocumentKey = std::string;
using ChunkKey = std::string;
using DocumentRevision = std::uint64_t;

struct MetadataFilter {
    using AllowedValues = std::vector<std::string>;
    using Requirements = std::map<std::string, AllowedValues>;

    // Keys are combined with AND and values within a key with OR. The map and
    // normalized vectors give canonical ordering for SQL binding and acks.
    Requirements required;
    bool supplied = false;

    MetadataFilter() = default;

    // Scalar compatibility: existing callers and the C ABI express one value
    // per key and are normalized into singleton sets.
    MetadataFilter(Metadata scalar) : supplied(true) {
        for (auto& [key, value] : scalar)
            required.emplace(std::move(key), AllowedValues{std::move(value)});
    }

    MetadataFilter(Requirements values) : required(std::move(values)), supplied(true) {
        for (auto& [key, allowed] : required) {
            std::sort(allowed.begin(), allowed.end());
            allowed.erase(std::unique(allowed.begin(), allowed.end()), allowed.end());
        }
    }

    [[nodiscard]] bool empty() const noexcept { return required.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return supplied; }

    [[nodiscard]] bool matches(const Metadata& actual) const noexcept {
        if (!supplied)
            return true;
        if (required.empty())
            return false;
        for (const auto& [key, allowed] : required) {
            const auto found = actual.find(key);
            if (found == actual.end() ||
                !std::binary_search(allowed.begin(), allowed.end(), found->second))
                return false;
        }
        return true;
    }

    friend bool operator==(const MetadataFilter&, const MetadataFilter&) = default;
};

struct PreparedChunk {
    ChunkKey key;
    std::size_t ordinal = 0;
    std::string text;
    std::string indexed_text;
    std::string context;
    std::uint32_t start_line = 0;
    std::uint32_t end_line = 0;
    Vector embedding;
};

struct PreparedDocument {
    DocumentKey key;
    std::string title;
    std::string content;
    Metadata metadata;
    std::string content_hash;
    std::string chunking_fingerprint;
    std::string embedding_identity;
    std::vector<PreparedChunk> chunks;
};

enum class SearchMode { lexical, dense, hybrid };

struct SearchRequest {
    std::string query;
    std::optional<Vector> embedding;
    SearchMode mode = SearchMode::hybrid;
    std::size_t top_k = 10;
    std::size_t candidate_pool = 60;
    MetadataFilter filter;
    std::string profile = "balanced";
};

enum class ScoreType { bm25, cosine };

struct Candidate {
    ChunkKey chunk;
    float raw_score = 0.0F;
    ScoreType score_type = ScoreType::cosine;

    friend bool operator==(const Candidate&, const Candidate&) = default;
};

using CandidateList = std::vector<Candidate>;

struct LexicalRequest {
    std::string_view query;
    std::size_t k = 10;
    MetadataFilter filter;
};

struct DenseRequest {
    VectorView query;
    std::size_t k = 10;
    MetadataFilter filter;
};

struct HybridRequest {
    LexicalRequest lexical;
    DenseRequest dense;
};

struct CandidateBatch {
    CandidateList lexical;
    CandidateList dense;
};

struct FetchOptions {
    bool include_text = true;
    bool include_embedding = false;
};

struct StoredChunk {
    ChunkKey key;
    DocumentKey document;
    DocumentRevision revision = 0;
    std::size_t ordinal = 0;
    std::string title;
    std::string text;
    std::string context;
    Metadata metadata;
    std::uint32_t start_line = 0;
    std::uint32_t end_line = 0;
    Vector embedding;
};

struct BackendCapabilities {
    bool lexical = false;
    bool dense = false;
    bool durable = false;
    bool concurrent_reads = false;
    bool atomic_document_activation = false;
};

struct BackendStats {
    std::size_t live_documents = 0;
    std::size_t live_chunks = 0;
    std::size_t embedding_dimension = 0;
    std::size_t dense_bytes = 0;
    std::size_t dense_mapped_bytes = 0;
    bool dense_exact = true;
    std::string dense_implementation = "native";
    std::string dense_algorithm = "exact";
    std::size_t catalog_embedding_bytes = 0;
    std::size_t dense_base_chunks = 0;
    std::size_t dense_delta_chunks = 0;
    std::size_t tombstones = 0;
    std::uint64_t generation = 0;
    std::size_t estimated_compaction_bytes = 0;
    bool maintenance_required = false;
    BackendCapabilities capabilities;
};

class CandidateBackend {
  public:
    virtual ~CandidateBackend() = default;

    virtual Result<void> activate(PreparedDocument document, DocumentRevision revision) = 0;
    virtual Result<bool> erase(DocumentKey document, DocumentRevision revision) = 0;

    [[nodiscard]] virtual Result<CandidateList>
    lexical_candidates(const LexicalRequest& request) const = 0;
    [[nodiscard]] virtual Result<CandidateList>
    dense_candidates(const DenseRequest& request) const = 0;

    // Backends with transactional snapshots override this operation. The
    // default preserves local parallel lexical/dense execution.
    [[nodiscard]] virtual Result<CandidateBatch>
    hybrid_candidates(const HybridRequest& request) const;

    [[nodiscard]] virtual Result<std::vector<StoredChunk>> fetch(std::span<const ChunkKey> chunks,
                                                                 FetchOptions options) const = 0;
    [[nodiscard]] virtual Result<BackendStats> stats() const = 0;
};

} // namespace rag::backend
