#pragma once

#include <cstddef>
#include <string>

#include "rag/backend/candidate_backend.hpp"
#include "rag/dense/embedder.hpp"
#include "rag/text/chunker.hpp"

namespace rag::preparation {

struct PrepareOptions {
    text::ChunkOptions chunking;
    std::size_t embedding_batch_size = 32;
};

// Pure preparation boundary: no storage is mutated until this returns a fully
// validated document. Passing no embedder creates a lexical-only document.
[[nodiscard]] Result<backend::PreparedDocument>
prepare_chunks(backend::DocumentKey key, std::string content, Metadata metadata = {},
               std::string title = {}, const text::ChunkOptions& options = {});

[[nodiscard]] Result<void> embed_document(backend::PreparedDocument& document,
                                          const dense::AnyEmbedder& embedder,
                                          std::size_t batch_size = 32);

[[nodiscard]] Result<backend::PreparedDocument>
prepare_document(backend::DocumentKey key, std::string content, Metadata metadata = {},
                 std::string title = {}, PrepareOptions options = {},
                 const dense::AnyEmbedder* embedder = nullptr);

} // namespace rag::preparation
