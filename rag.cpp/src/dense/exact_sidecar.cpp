#include "rag/dense/exact_sidecar.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <new>
#include <unordered_set>

#include "rag/dense/cache_fingerprint.hpp"
#include "rag/dense/simd.hpp"
#include "rag/store/container.hpp"
#include "rag/store/format.hpp"

namespace rag::dense {
namespace {

constexpr std::uint32_t kSidecarVersion = 1;
constexpr std::string_view kImplementation = "native";
constexpr std::string_view kAlgorithm = "exact";

bool normalized(VectorView vector) {
    if (vector.empty())
        return false;
    double norm = 0.0;
    for (const float value : vector) {
        if (!std::isfinite(value))
            return false;
        norm += static_cast<double>(value) * value;
    }
    return std::abs(norm - 1.0) <= 1.0e-3;
}

} // namespace

std::string exact_sidecar_path(std::string_view checkpoint_path) {
    return std::string(checkpoint_path) + ".dense.native-exact-v1";
}

std::string exact_sidecar_fingerprint(VectorSource source, std::string_view embedding_identity) {
    return dense_cache_fingerprint(source, embedding_identity,
                                   {"native", "exact", kSidecarVersion, {}});
}

Result<void> write_exact_sidecar(const std::string& path, std::string_view fingerprint,
                                 VectorSource source) {
    try {
        if (fingerprint.empty())
            return fail<void>(Errc::invalid_argument, "dense sidecar fingerprint is required");
        const std::size_t dimension = source.empty() ? 0 : source.front().vector.size();
        store::Writer metadata;
        metadata.u<std::uint32_t>(kSidecarVersion);
        metadata.str(kImplementation);
        metadata.str(kAlgorithm);
        metadata.str(fingerprint);
        metadata.u<std::uint64_t>(source.size());
        metadata.u<std::uint64_t>(dimension);

        store::Writer keys;
        keys.u<std::uint64_t>(source.size());
        store::Writer vectors;
        std::unordered_set<backend::ChunkKey> unique;
        for (const auto& row : source) {
            if (row.key.empty() || !unique.emplace(row.key).second)
                return fail<void>(Errc::invalid_argument, "dense sidecar keys are invalid");
            if (row.vector.size() != dimension || !normalized(row.vector))
                return fail<void>(Errc::invalid_argument, "dense sidecar vector is invalid");
            keys.str(row.key);
            vectors.bytes(std::string_view(reinterpret_cast<const char*>(row.vector.data()),
                                           row.vector.size() * sizeof(float)));
        }

        store::Container container;
        container.put(store::Tag::meta, std::move(metadata.data()));
        container.put(store::Tag::chunks, std::move(keys.data()));
        container.put(store::Tag::embed, std::move(vectors.data()));
        container.set_flags(store::kHasEmbeddings);
        return container.write_file(path);
    } catch (const std::exception& error) {
        return fail<void>(Errc::io_error,
                          "dense sidecar write failed: " + std::string(error.what()));
    } catch (...) {
        return fail<void>(Errc::io_error, "dense sidecar write failed");
    }
}

Result<std::shared_ptr<MappedExactIndex>>
MappedExactIndex::open(const std::string& path, std::string_view expected_fingerprint) {
    try {
        auto mapping = store::ContainerView::open_file(path);
        if (!mapping)
            return unexpected(mapping.error());
        const auto metadata_blob = mapping->get(store::Tag::meta);
        const auto keys_blob = mapping->get(store::Tag::chunks);
        const auto vectors_blob = mapping->get(store::Tag::embed);
        if (!metadata_blob || !keys_blob || !vectors_blob ||
            (mapping->flags() & store::kHasEmbeddings) == 0)
            return fail<std::shared_ptr<MappedExactIndex>>(Errc::corrupt_index,
                                                           "dense sidecar sections are invalid");

        store::Reader metadata(*metadata_blob);
        std::uint32_t version = 0;
        std::string implementation;
        std::string algorithm;
        std::string fingerprint;
        std::uint64_t rows = 0;
        std::uint64_t dimension = 0;
        if (!metadata.u(version) || version != kSidecarVersion || !metadata.str(implementation) ||
            implementation != kImplementation || !metadata.str(algorithm) ||
            algorithm != kAlgorithm || !metadata.str(fingerprint) ||
            fingerprint != expected_fingerprint || !metadata.u(rows) || !metadata.u(dimension) ||
            metadata.remaining() != 0 || rows > store::kMaxChunks ||
            dimension > store::kMaxVectorDimension || (rows != 0 && dimension == 0))
            return fail<std::shared_ptr<MappedExactIndex>>(Errc::corrupt_index,
                                                           "dense sidecar metadata is invalid");
        if (dimension != 0 && rows > vectors_blob->size() / sizeof(float) / dimension)
            return fail<std::shared_ptr<MappedExactIndex>>(Errc::corrupt_index,
                                                           "dense sidecar matrix is truncated");
        if (vectors_blob->size() != rows * dimension * sizeof(float))
            return fail<std::shared_ptr<MappedExactIndex>>(Errc::corrupt_index,
                                                           "dense sidecar matrix size is invalid");

        store::Reader keys_reader(*keys_blob);
        std::uint64_t key_count = 0;
        if (!keys_reader.u(key_count) || key_count != rows)
            return fail<std::shared_ptr<MappedExactIndex>>(Errc::corrupt_index,
                                                           "dense sidecar key count is invalid");
        std::vector<backend::ChunkKey> keys;
        keys.reserve(static_cast<std::size_t>(rows));
        std::unordered_set<backend::ChunkKey> unique;
        Vector vector(static_cast<std::size_t>(dimension));
        for (std::uint64_t row = 0; row < rows; ++row) {
            backend::ChunkKey key;
            if (!keys_reader.str(key) || key.empty() || !unique.insert(key).second)
                return fail<std::shared_ptr<MappedExactIndex>>(Errc::corrupt_index,
                                                               "dense sidecar key is invalid");
            keys.push_back(std::move(key));
            std::memcpy(vector.data(), vectors_blob->data() + row * dimension * sizeof(float),
                        vector.size() * sizeof(float));
            if (!normalized(vector))
                return fail<std::shared_ptr<MappedExactIndex>>(Errc::corrupt_index,
                                                               "dense sidecar vector is invalid");
        }
        if (keys_reader.remaining() != 0)
            return fail<std::shared_ptr<MappedExactIndex>>(Errc::corrupt_index,
                                                           "dense sidecar has trailing keys");

        auto index = std::make_shared<MappedExactIndex>();
        index->mapping_ = std::move(*mapping);
        index->keys_ = std::move(keys);
        index->vectors_ = *vectors_blob;
        index->dimension_ = static_cast<std::size_t>(dimension);
        return index;
    } catch (const std::bad_alloc&) {
        return fail<std::shared_ptr<MappedExactIndex>>(Errc::corrupt_index,
                                                       "dense sidecar allocation failed");
    } catch (const std::exception& error) {
        return fail<std::shared_ptr<MappedExactIndex>>(
            Errc::corrupt_index, "dense sidecar load failed: " + std::string(error.what()));
    } catch (...) {
        return fail<std::shared_ptr<MappedExactIndex>>(Errc::corrupt_index,
                                                       "dense sidecar load failed");
    }
}

Result<void> MappedExactIndex::build(VectorSource) {
    return fail<void>(Errc::invalid_argument, "mapped dense sidecars are read-only");
}

Result<backend::CandidateList> MappedExactIndex::search(Vector query, AllowedIds allowed,
                                                        std::size_t k) const {
    if (k == 0 || keys_.empty())
        return backend::CandidateList{};
    if (query.size() != dimension_)
        return fail<backend::CandidateList>(Errc::dimension_mismatch,
                                            "query dimension does not match dense sidecar");
    if (!normalized(query))
        return fail<backend::CandidateList>(Errc::invalid_argument,
                                            "dense query is not unit normalized");
    std::unordered_set<backend::ChunkKey> allowed_set;
    if (!allowed.ids.empty())
        allowed_set.insert(allowed.ids.begin(), allowed.ids.end());

    backend::CandidateList candidates;
    candidates.reserve(keys_.size());
    Vector row(dimension_);
    for (std::size_t index = 0; index < keys_.size(); ++index) {
        if (!allowed_set.empty() && !allowed_set.contains(keys_[index]))
            continue;
        std::memcpy(row.data(), vectors_.data() + index * dimension_ * sizeof(float),
                    row.size() * sizeof(float));
        candidates.push_back({keys_[index], dot(query, row), backend::ScoreType::cosine});
    }
    const auto order = [](const backend::Candidate& left, const backend::Candidate& right) {
        if (left.raw_score != right.raw_score)
            return left.raw_score > right.raw_score;
        return left.chunk < right.chunk;
    };
    const auto wanted = std::min(k, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(wanted),
                      candidates.end(), order);
    candidates.resize(wanted);
    return candidates;
}

DenseIndexStats MappedExactIndex::stats() const {
    std::size_t resident = 0;
    for (const auto& key : keys_)
        resident += key.capacity();
    DenseIndexStats output;
    output.vectors = keys_.size();
    output.dimension = dimension_;
    output.resident_bytes = resident;
    output.exact = true;
    output.implementation = "native";
    output.algorithm = "exact";
    output.mapped_bytes = mapping_.mapped_bytes();
    return output;
}

} // namespace rag::dense
