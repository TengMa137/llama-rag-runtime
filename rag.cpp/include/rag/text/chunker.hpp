#pragma once
// rag/text/chunker.hpp — semantic, line-aligned document chunking.
//
// Splits a document into bounded chunks that (a) never break mid-line, (b)
// prefer to break on blank lines / markdown headings (semantic boundaries),
// (c) overlap a few trailing lines so a fact straddling a boundary survives,
// and (d) carry a `context` breadcrumb (the enclosing markdown heading chain)
// for contextual retrieval.

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/document.hpp"

namespace rag::text {

enum class TokenCountingMode { exact, conservative_utf8_bytes };
enum class InvalidUtf8Policy { replace, reject };

using TokenMeasurer = std::function<std::size_t(std::string_view)>;

struct EmbeddingPolicy {
    std::string model_identity;
    std::string tokenizer_identity;
    std::size_t dimension = 0;
    std::size_t target_tokens = 0;
    std::size_t max_tokens = 0;
    std::size_t overlap_tokens = 0;
    std::size_t reserved_tokens = 0;
    std::string document_prefix;
    std::string query_prefix;
    TokenCountingMode counting_mode = TokenCountingMode::conservative_utf8_bytes;
    InvalidUtf8Policy invalid_utf8 = InvalidUtf8Policy::replace;
};

struct ChunkOptions {
    // Legacy v1 values remain readable and are used when policy.max_tokens is
    // zero. New indexes should set policy and a model-specific measurer.
    std::size_t max_lines = 40;
    std::size_t max_chars = 1600;
    std::size_t overlap_lines = 4;
    bool heading_context = true; // synthesize breadcrumb from headings
    EmbeddingPolicy policy{};
    TokenMeasurer measure_tokens{};
};

[[nodiscard]] std::string chunking_fingerprint(const ChunkOptions& opts);

// Split `body` into chunks. Chunk ids are left invalid (the Corpus assigns
// them on ingest); doc is set to `doc_id`.
[[nodiscard]] std::vector<Chunk> chunk_document(DocId doc_id, const std::string& body,
                                                const ChunkOptions& opts = {});

} // namespace rag::text
