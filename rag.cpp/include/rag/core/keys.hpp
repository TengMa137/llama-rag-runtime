#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "rag/core/document.hpp"

namespace rag {

// Canonical newline normalization used before chunking, hashing, and stable ID
// generation. It deliberately does not alter any other source bytes.
[[nodiscard]] std::string normalize_source_text(std::string_view input);

[[nodiscard]] std::string stable_chunk_key(std::string_view document_key, std::uint32_t start_line,
                                           std::uint32_t end_line, std::string_view text);

[[nodiscard]] std::string document_content_hash(std::string_view title,
                                                std::string_view normalized_content,
                                                const Metadata& metadata);

} // namespace rag
