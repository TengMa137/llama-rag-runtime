#include "rag/preparation/document_preparer.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <utility>
#include <vector>

#include "rag/core/keys.hpp"
#include "rag/dense/simd.hpp"

namespace rag::preparation {

Result<backend::PreparedDocument> prepare_chunks(backend::DocumentKey key, std::string content,
                                                 Metadata metadata, std::string title,
                                                 const text::ChunkOptions& options) {
    if (key.empty())
        return fail<backend::PreparedDocument>(Errc::invalid_argument,
                                               "document key must not be empty");
    if (content.empty())
        return fail<backend::PreparedDocument>(Errc::invalid_argument,
                                               "document content must not be empty");

    backend::PreparedDocument prepared;
    prepared.key = std::move(key);
    prepared.title = std::move(title);
    prepared.content = normalize_source_text(content);
    prepared.metadata = std::move(metadata);
    prepared.content_hash =
        document_content_hash(prepared.title, prepared.content, prepared.metadata);
    prepared.chunking_fingerprint = text::chunking_fingerprint(options);

    auto chunks = text::chunk_document(DocId{0}, prepared.content, options);
    prepared.chunks.reserve(chunks.size());
    for (std::size_t ordinal = 0; ordinal < chunks.size(); ++ordinal) {
        auto& chunk = chunks[ordinal];
        backend::PreparedChunk output;
        output.key = stable_chunk_key(prepared.key, chunk.start_line, chunk.end_line, chunk.text);
        output.ordinal = ordinal;
        output.text = std::move(chunk.text);
        output.context = std::move(chunk.context);
        output.indexed_text =
            output.context.empty() ? output.text : output.context + "\n" + output.text;
        output.start_line = chunk.start_line;
        output.end_line = chunk.end_line;
        prepared.chunks.push_back(std::move(output));
    }

    return prepared;
}

Result<void> embed_document(backend::PreparedDocument& prepared, const dense::AnyEmbedder& embedder,
                            std::size_t batch_size) {
    if (prepared.chunks.empty()) {
        prepared.embedding_identity = std::string(embedder.identity());
        return {};
    }
    batch_size = batch_size ? batch_size : 1;
    std::vector<std::string> embedding_texts;
    embedding_texts.reserve(prepared.chunks.size());
    for (const auto& chunk : prepared.chunks)
        embedding_texts.push_back(chunk.indexed_text);
    for (std::size_t offset = 0; offset < embedding_texts.size(); offset += batch_size) {
        const std::size_t count = std::min(batch_size, embedding_texts.size() - offset);
        const std::span<const std::string> batch(embedding_texts.data() + offset, count);
        auto vectors = embedder.embed(batch);
        if (!vectors)
            return unexpected(vectors.error());
        if (vectors->size() != count)
            return fail<void>(Errc::transport_error, "embedding result count does not match input");
        for (std::size_t index = 0; index < count; ++index) {
            auto& vector = (*vectors)[index];
            if (vector.size() != embedder.dimension())
                return fail<void>(Errc::dimension_mismatch,
                                  "embedding dimension does not match policy");
            double norm = 0.0;
            for (const float value : vector) {
                if (!std::isfinite(value))
                    return fail<void>(Errc::invalid_argument,
                                      "embedding contains a non-finite value");
                norm += static_cast<double>(value) * static_cast<double>(value);
            }
            if (norm <= 0.0)
                return fail<void>(Errc::invalid_argument, "embedding has zero norm");
            dense::normalize(vector);
            prepared.chunks[offset + index].embedding = std::move(vector);
        }
    }
    prepared.embedding_identity = std::string(embedder.identity());
    return {};
}

Result<backend::PreparedDocument> prepare_document(backend::DocumentKey key, std::string content,
                                                   Metadata metadata, std::string title,
                                                   PrepareOptions options,
                                                   const dense::AnyEmbedder* embedder) {
    auto prepared = prepare_chunks(std::move(key), std::move(content), std::move(metadata),
                                   std::move(title), options.chunking);
    if (!prepared || !embedder)
        return prepared;
    if (auto embedded = embed_document(*prepared, *embedder, options.embedding_batch_size);
        !embedded)
        return unexpected(embedded.error());
    return prepared;
}

} // namespace rag::preparation
