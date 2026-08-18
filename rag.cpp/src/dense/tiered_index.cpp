#include "rag/dense/tiered_index.hpp"

#include <algorithm>
#include <unordered_set>

namespace rag::dense {

struct TieredDenseIndex::Base {
    std::shared_ptr<DenseIndex> index = std::make_shared<NativeExactIndex>();
    std::unordered_set<backend::ChunkKey> keys;
    std::size_t dimension = 0;
};

Result<std::shared_ptr<const TieredDenseIndex>> TieredDenseIndex::empty() {
    auto tiered = std::make_shared<TieredDenseIndex>();
    auto base = std::make_shared<Base>();
    if (auto built = base->index->build({}); !built)
        return unexpected(built.error());
    tiered->base_ = std::move(base);
    tiered->delta_ = std::make_shared<NativeExactIndex>();
    if (auto built = tiered->delta_->build({}); !built)
        return unexpected(built.error());
    return std::shared_ptr<const TieredDenseIndex>(std::move(tiered));
}

Result<std::shared_ptr<const TieredDenseIndex>>
TieredDenseIndex::with_delta(VectorSource live_delta,
                             std::unordered_set<backend::ChunkKey> tombstones,
                             std::size_t dimension) const {
    if (base_->dimension != 0 && dimension != 0 && base_->dimension != dimension)
        return fail<std::shared_ptr<const TieredDenseIndex>>(
            Errc::dimension_mismatch, "dense delta dimension is incompatible with base");
    for (const auto& key : tombstones)
        if (!base_->keys.contains(key))
            return fail<std::shared_ptr<const TieredDenseIndex>>(
                Errc::invalid_argument, "dense tombstone does not belong to base");
    for (const auto& row : live_delta) {
        if (row.vector.size() != dimension)
            return fail<std::shared_ptr<const TieredDenseIndex>>(
                Errc::dimension_mismatch, "dense delta row dimension is invalid");
        const backend::ChunkKey key(row.key);
        if (base_->keys.contains(key) && !tombstones.contains(key))
            return fail<std::shared_ptr<const TieredDenseIndex>>(
                Errc::already_exists, "dense row is already represented by base");
    }
    auto tiered = std::make_shared<TieredDenseIndex>();
    tiered->base_ = base_;
    tiered->delta_ = std::make_shared<NativeExactIndex>();
    if (auto built = tiered->delta_->build(live_delta); !built)
        return unexpected(built.error());
    tiered->delta_keys_.reserve(live_delta.size());
    for (const auto& row : live_delta)
        tiered->delta_keys_.emplace(row.key);
    tiered->tombstones_ = std::move(tombstones);
    tiered->dimension_ = dimension;
    return std::shared_ptr<const TieredDenseIndex>(std::move(tiered));
}

Result<std::shared_ptr<const TieredDenseIndex>>
TieredDenseIndex::compact(VectorSource live_vectors, std::size_t dimension,
                          const DensePolicy& policy) const {
    auto tiered = std::make_shared<TieredDenseIndex>();
    auto base = std::make_shared<Base>();
    base->dimension = dimension;
    for (const auto& row : live_vectors)
        if (row.vector.size() != dimension)
            return fail<std::shared_ptr<const TieredDenseIndex>>(
                Errc::dimension_mismatch, "dense base row dimension is invalid");
    auto built = build_dense_index(policy, live_vectors);
    if (!built)
        return unexpected(built.error());
    base->index = std::move(*built);
    base->keys.reserve(live_vectors.size());
    for (const auto& row : live_vectors)
        base->keys.emplace(row.key);
    tiered->base_ = std::move(base);
    tiered->delta_ = std::make_shared<NativeExactIndex>();
    if (auto built = tiered->delta_->build({}); !built)
        return unexpected(built.error());
    tiered->dimension_ = dimension;
    return std::shared_ptr<const TieredDenseIndex>(std::move(tiered));
}

Result<std::shared_ptr<const TieredDenseIndex>>
TieredDenseIndex::from_base(std::shared_ptr<DenseIndex> index,
                            std::span<const backend::ChunkKey> keys, std::size_t dimension) {
    if (!index)
        return fail<std::shared_ptr<const TieredDenseIndex>>(Errc::invalid_argument,
                                                             "dense base index is required");
    const auto stats = index->stats();
    if (stats.vectors != keys.size() || stats.dimension != dimension)
        return fail<std::shared_ptr<const TieredDenseIndex>>(
            Errc::dimension_mismatch, "dense base index does not match its key catalog");
    auto base = std::make_shared<Base>();
    base->index = std::move(index);
    base->dimension = dimension;
    base->keys.reserve(keys.size());
    for (const auto& key : keys)
        if (key.empty() || !base->keys.insert(key).second)
            return fail<std::shared_ptr<const TieredDenseIndex>>(
                Errc::invalid_argument, "dense base key catalog is invalid");
    auto tiered = std::make_shared<TieredDenseIndex>();
    tiered->base_ = std::move(base);
    tiered->delta_ = std::make_shared<NativeExactIndex>();
    if (auto built = tiered->delta_->build({}); !built)
        return unexpected(built.error());
    tiered->dimension_ = dimension;
    return std::shared_ptr<const TieredDenseIndex>(std::move(tiered));
}

Result<backend::CandidateList>
TieredDenseIndex::search(VectorView query, std::span<const backend::ChunkKey> base_allowed,
                         std::span<const backend::ChunkKey> delta_allowed, std::size_t k) const {
    backend::CandidateList merged;
    const Vector owned_query(query.begin(), query.end());
    if (!base_allowed.empty()) {
        auto base = base_->index->search(owned_query, {base_allowed}, k);
        if (!base)
            return unexpected(base.error());
        merged.insert(merged.end(), base->begin(), base->end());
    }
    if (!delta_allowed.empty()) {
        auto delta = delta_->search(owned_query, {delta_allowed}, k);
        if (!delta)
            return unexpected(delta.error());
        merged.insert(merged.end(), delta->begin(), delta->end());
    }
    std::sort(merged.begin(), merged.end(), [](const auto& left, const auto& right) {
        if (left.raw_score != right.raw_score)
            return left.raw_score > right.raw_score;
        return left.chunk < right.chunk;
    });
    backend::CandidateList output;
    output.reserve(std::min(k, merged.size()));
    std::unordered_set<backend::ChunkKey> seen;
    for (auto& candidate : merged) {
        if (!seen.insert(candidate.chunk).second)
            continue;
        output.push_back(std::move(candidate));
        if (output.size() == k)
            break;
    }
    return output;
}

bool TieredDenseIndex::contains_base(const backend::ChunkKey& key) const {
    return base_->keys.contains(key) && !tombstones_.contains(key);
}

bool TieredDenseIndex::contains_delta(const backend::ChunkKey& key) const {
    return delta_keys_.contains(key);
}

std::shared_ptr<const DenseIndex> TieredDenseIndex::base_index() const noexcept {
    return base_->index;
}

TieredIndexStats TieredDenseIndex::stats() const {
    const auto base_stats = base_->index->stats();
    return {base_->keys.size(),
            delta_keys_.size(),
            tombstones_.size(),
            dimension_,
            base_stats.resident_bytes + delta_->stats().resident_bytes,
            base_stats.mapped_bytes,
            base_stats.exact,
            base_stats.implementation,
            base_stats.algorithm};
}

} // namespace rag::dense
