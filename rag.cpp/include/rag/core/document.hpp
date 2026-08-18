#pragma once
// rag/core/document.hpp — the corpus record types (product types).

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "rag/core/types.hpp"

namespace rag {

// Arbitrary user metadata attached to a document. Used for filtered retrieval.
using Metadata = std::map<std::string, std::string>;

// A source document as ingested by the user. `id` is assigned by the Corpus.
struct Document {
    DocId id = DocId::invalid();
    std::string uri;   // stable external identity (path, URL, key)
    std::string title; // optional display title / heading breadcrumb root
    std::string text;  // full body
    Metadata meta;     // filterable key/value pairs
};

// A chunk is a bounded, retrievable window into a Document. Its structural
// heading breadcrumb is prepended to the indexed text when enabled.
struct Chunk {
    ChunkId id = ChunkId::invalid();
    DocId doc = DocId::invalid();
    std::string text;    // the chunk body (raw)
    std::string context; // structural heading breadcrumb
    std::uint32_t start_line = 0;
    std::uint32_t end_line = 0;
    Vector embedding;               // dense vector (empty until embedded)
    const Metadata* meta = nullptr; // borrowed from the owning Document

    // The text actually fed to the lexical + dense indexes: context ⊕ body.
    [[nodiscard]] std::string indexed_text() const {
        if (context.empty())
            return text;
        return context + "\n" + text;
    }
};

// A ranked, resolved retrieval result the caller consumes.
struct SearchResult {
    ChunkId chunk = ChunkId::invalid();
    DocId doc = DocId::invalid();
    std::string chunk_key;    // stable backend-neutral public ID
    std::string document_key; // stable caller-supplied document ID
    std::uint64_t revision = 0;
    Score score{0.0f};
    std::string text;
    std::string context;
    std::string uri;
    std::string title;
    Metadata metadata;
    std::uint32_t start_line = 0;
    std::uint32_t end_line = 0;
};

} // namespace rag
