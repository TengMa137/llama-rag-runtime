#include "rag/dense/cache_fingerprint.hpp"

namespace rag::dense {
namespace {

void hash_bytes(std::uint64_t& hash, std::string_view bytes) {
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
}

void hash_field(std::uint64_t& hash, std::string_view value) {
    const auto size = std::to_string(value.size());
    hash_bytes(hash, size);
    hash_bytes(hash, ":");
    hash_bytes(hash, value);
    hash_bytes(hash, "\n");
}

std::string hex64(std::uint64_t value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string output(16, '0');
    for (int index = 15; index >= 0; --index) {
        output[static_cast<std::size_t>(index)] = digits[value & 0xFU];
        value >>= 4U;
    }
    return output;
}

} // namespace

std::string dense_cache_fingerprint(VectorSource source, std::string_view embedding_identity,
                                    const DenseCacheIdentity& identity) {
    std::uint64_t hash = 1469598103934665603ULL;
    hash_field(hash, identity.implementation);
    hash_field(hash, identity.algorithm);
    hash_field(hash, std::to_string(identity.version));
    hash_field(hash, identity.parameters);
    hash_field(hash, embedding_identity);
    hash_field(hash, std::to_string(source.size()));
    const std::size_t dimension = source.empty() ? 0 : source.front().vector.size();
    hash_field(hash, std::to_string(dimension));
    for (const auto& row : source) {
        hash_field(hash, row.key);
        hash_field(hash, std::string_view(reinterpret_cast<const char*>(row.vector.data()),
                                          row.vector.size() * sizeof(float)));
    }
    return "fnv1a64:" + hex64(hash);
}

} // namespace rag::dense
