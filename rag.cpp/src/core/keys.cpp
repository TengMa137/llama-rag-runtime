#include "rag/core/keys.hpp"

namespace rag {
namespace {

std::uint64_t fnv1a(std::string_view text, std::uint64_t hash = 1469598103934665603ULL) {
    for (const unsigned char value : text) {
        hash ^= value;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string hex64(std::uint64_t value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string encoded(16, '0');
    for (int index = 15; index >= 0; --index) {
        encoded[static_cast<std::size_t>(index)] = digits[value & 0xFU];
        value >>= 4U;
    }
    return encoded;
}

void hash_field(std::uint64_t& hash, std::string_view value) {
    hash = fnv1a(std::to_string(value.size()), hash);
    hash = fnv1a(":", hash);
    hash = fnv1a(value, hash);
    hash = fnv1a("\n", hash);
}

} // namespace

std::string normalize_source_text(std::string_view input) {
    std::string output;
    output.reserve(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        if (input[index] != '\r') {
            output.push_back(input[index]);
            continue;
        }
        if (index + 1 < input.size() && input[index + 1] == '\n')
            continue;
        output.push_back('\n');
    }
    return output;
}

std::string stable_chunk_key(std::string_view document_key, std::uint32_t start_line,
                             std::uint32_t end_line, std::string_view text) {
    std::string identity;
    identity.reserve(document_key.size() + text.size() + 32);
    identity.append(document_key);
    identity.push_back('\n');
    identity.append(std::to_string(start_line));
    identity.push_back(':');
    identity.append(std::to_string(end_line));
    identity.push_back('\n');
    identity.append(normalize_source_text(text));
    return "chk_" + hex64(fnv1a(identity));
}

std::string document_content_hash(std::string_view title, std::string_view normalized_content,
                                  const Metadata& metadata) {
    std::uint64_t hash = 1469598103934665603ULL;
    hash_field(hash, title);
    hash_field(hash, normalized_content);
    for (const auto& [key, value] : metadata) {
        hash_field(hash, key);
        hash_field(hash, value);
    }
    return "fnv1a64:" + hex64(hash);
}

} // namespace rag
