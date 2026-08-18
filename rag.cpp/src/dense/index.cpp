#include "rag/dense/index.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <unordered_set>

#include "rag/dense/simd.hpp"

namespace rag::dense {
namespace {

constexpr float kUnitTolerance = 1.0e-3F;

bool normalized(VectorView vector) {
    if (vector.empty())
        return false;
    double norm = 0.0;
    for (const float value : vector) {
        if (!std::isfinite(value))
            return false;
        norm += static_cast<double>(value) * static_cast<double>(value);
    }
    return std::abs(norm - 1.0) <= kUnitTolerance;
}

} // namespace

bool AllowedIds::allows(const backend::ChunkKey& key) const noexcept {
    return ids.empty() || std::find(ids.begin(), ids.end(), key) != ids.end();
}

Result<void> NativeExactIndex::build(VectorSource source) {
    std::size_t dimension = 0;
    std::vector<backend::ChunkKey> keys;
    std::vector<float> vectors;
    std::unordered_set<backend::ChunkKey> unique;

    keys.reserve(source.size());
    if (!source.empty()) {
        dimension = source.front().vector.size();
        if (dimension == 0)
            return fail<void>(Errc::invalid_argument, "dense vectors must not be empty");
        vectors.reserve(source.size() * dimension);
    }

    for (const auto& row : source) {
        if (row.key.empty())
            return fail<void>(Errc::invalid_argument, "dense vector key must not be empty");
        if (!unique.emplace(row.key).second)
            return fail<void>(Errc::already_exists, "duplicate dense vector key");
        if (row.vector.size() != dimension)
            return fail<void>(Errc::dimension_mismatch, "dense vector dimensions differ");
        if (!normalized(row.vector))
            return fail<void>(Errc::invalid_argument, "dense vector is not unit normalized");
        keys.emplace_back(row.key);
        vectors.insert(vectors.end(), row.vector.begin(), row.vector.end());
    }

    std::unique_lock lock(mutex_);
    keys_ = std::move(keys);
    vectors_ = std::move(vectors);
    dimension_ = dimension;
    return {};
}

Result<backend::CandidateList> NativeExactIndex::search(Vector query, AllowedIds allowed,
                                                        std::size_t k) const {
    std::shared_lock lock(mutex_);
    if (k == 0 || keys_.empty())
        return backend::CandidateList{};
    if (query.size() != dimension_)
        return fail<backend::CandidateList>(Errc::dimension_mismatch,
                                            "query dimension does not match dense index");
    if (!normalized(query))
        return fail<backend::CandidateList>(Errc::invalid_argument,
                                            "dense query is not unit normalized");

    std::unordered_set<backend::ChunkKey> allowed_set;
    if (!allowed.ids.empty())
        allowed_set.insert(allowed.ids.begin(), allowed.ids.end());

    backend::CandidateList candidates;
    candidates.reserve(keys_.size());
    for (std::size_t row = 0; row < keys_.size(); ++row) {
        if (!allowed_set.empty() && !allowed_set.contains(keys_[row]))
            continue;
        const VectorView vector(vectors_.data() + row * dimension_, dimension_);
        candidates.push_back({keys_[row], dot(query, vector), backend::ScoreType::cosine});
    }

    const auto order = [](const backend::Candidate& left, const backend::Candidate& right) {
        if (left.raw_score != right.raw_score)
            return left.raw_score > right.raw_score;
        return left.chunk < right.chunk;
    };
    const std::size_t wanted = std::min(k, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(wanted),
                      candidates.end(), order);
    candidates.resize(wanted);
    return candidates;
}

DenseIndexStats NativeExactIndex::stats() const {
    std::shared_lock lock(mutex_);
    std::size_t bytes = vectors_.capacity() * sizeof(float);
    for (const auto& key : keys_)
        bytes += key.capacity();
    DenseIndexStats output;
    output.vectors = keys_.size();
    output.dimension = dimension_;
    output.resident_bytes = bytes;
    output.primary_vector_bytes = vectors_.capacity() * sizeof(float);
    output.exact = true;
    output.implementation = "native";
    output.algorithm = "exact";
    return output;
}

} // namespace rag::dense
