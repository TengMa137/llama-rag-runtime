#include "rag/dense/native_hnsw.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace rag::dense {
namespace {

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

NativeHnswIndex::NativeHnswIndex(NativeHnswPolicy policy) : policy_(policy) {}

Result<std::shared_ptr<NativeHnswIndex>>
NativeHnswIndex::from_serialized(std::string_view blob, std::span<const backend::ChunkKey> keys,
                                 NativeHnswPolicy policy) {
    if (auto valid = validate_policy(policy); !valid)
        return unexpected(valid.error());
    auto graph = index::HnswIndex::deserialize(blob);
    if (!graph)
        return unexpected(graph.error());
    const auto& config = graph->config();
    if (config.M != policy.neighbors || config.ef_construction != policy.ef_construction ||
        config.ef_search != policy.ef_search || config.seed != policy.seed || config.binary ||
        config.pq_codes != 0 || config.drop_floats || config.matryoshka_dim != graph->dimension() ||
        graph->size() != keys.size() || !graph->has_sequential_live_ids())
        return fail<std::shared_ptr<NativeHnswIndex>>(Errc::corrupt_index,
                                                      "native HNSW cache policy is invalid");
    const auto memory = graph->memory_use();
    if (memory.sq8 != 0 || memory.pq != 0)
        return fail<std::shared_ptr<NativeHnswIndex>>(Errc::corrupt_index,
                                                      "native HNSW cache has duplicate vectors");
    std::unordered_set<backend::ChunkKey> unique;
    for (const auto& key : keys)
        if (key.empty() || !unique.insert(key).second)
            return fail<std::shared_ptr<NativeHnswIndex>>(Errc::corrupt_index,
                                                          "native HNSW cache keys are invalid");

    auto output = std::make_shared<NativeHnswIndex>(policy);
    output->graph_ = std::make_unique<index::HnswIndex>(std::move(*graph));
    output->keys_.assign(keys.begin(), keys.end());
    output->dimension_ = output->graph_->dimension();
    return output;
}

Result<void> NativeHnswIndex::validate_policy(const NativeHnswPolicy& policy) {
    if (policy.neighbors < 2 || policy.neighbors > 128 || policy.ef_construction < 4 ||
        policy.ef_construction < policy.neighbors || policy.ef_search == 0)
        return fail<void>(Errc::invalid_argument, "native HNSW policy is invalid");
    return {};
}

Result<void> NativeHnswIndex::build(VectorSource source) {
    if (auto valid = validate_policy(policy_); !valid)
        return valid;
    if (source.size() >= ChunkId::invalid().get())
        return fail<void>(Errc::invalid_argument, "native HNSW row limit exceeded");
    std::size_t dimension = source.empty() ? 0 : source.front().vector.size();
    std::vector<backend::ChunkKey> keys;
    keys.reserve(source.size());
    std::unordered_set<backend::ChunkKey> unique;
    for (const auto& row : source) {
        if (row.key.empty() || !unique.emplace(row.key).second)
            return fail<void>(Errc::invalid_argument, "native HNSW keys are invalid");
        if (row.vector.size() != dimension)
            return fail<void>(Errc::dimension_mismatch, "native HNSW dimensions differ");
        if (!normalized(row.vector))
            return fail<void>(Errc::invalid_argument, "native HNSW vector is not normalized");
        keys.emplace_back(row.key);
    }

    index::HnswConfig config;
    config.M = policy_.neighbors;
    config.ef_construction = policy_.ef_construction;
    config.ef_search = policy_.ef_search;
    config.seed = policy_.seed;
    // A full-length Matryoshka walk is ordinary float cosine but disables the
    // legacy graph's automatic SQ8 mirror. The graph therefore owns exactly
    // one vector representation and can exact-scan it for selective filters.
    config.matryoshka_dim = dimension;
    auto graph = std::make_unique<index::HnswIndex>(config);
    graph->build_batch(
        source.size(), [&](std::size_t row) { return source[row].vector; },
        [](std::size_t row) { return static_cast<std::uint32_t>(row); });
    if (!source.empty())
        (void)graph->search(source.front().vector, 1);

    std::unique_lock lock(mutex_);
    graph_ = std::move(graph);
    keys_ = std::move(keys);
    dimension_ = dimension;
    return {};
}

Result<backend::CandidateList> NativeHnswIndex::search(Vector query, AllowedIds allowed,
                                                       std::size_t k) const {
    std::shared_lock lock(mutex_);
    if (k == 0 || keys_.empty())
        return backend::CandidateList{};
    if (!graph_)
        return fail<backend::CandidateList>(Errc::unavailable, "native HNSW is not built");
    if (query.size() != dimension_)
        return fail<backend::CandidateList>(Errc::dimension_mismatch,
                                            "query dimension does not match native HNSW");
    if (!normalized(query))
        return fail<backend::CandidateList>(Errc::invalid_argument,
                                            "native HNSW query is not normalized");

    if (allowed.ids.size() == keys_.size() &&
        std::equal(allowed.ids.begin(), allowed.ids.end(), keys_.begin())) {
        auto hits = graph_->search(query, std::min(k, keys_.size()));
        backend::CandidateList output;
        output.reserve(hits.size());
        for (const auto& hit : hits) {
            const auto row = hit.chunk.get();
            if (row >= keys_.size())
                return fail<backend::CandidateList>(Errc::corrupt_index,
                                                    "native HNSW returned an invalid row");
            output.push_back({keys_[row], hit.score.get(), backend::ScoreType::cosine});
        }
        std::sort(output.begin(), output.end(), [](const auto& left, const auto& right) {
            if (left.raw_score != right.raw_score)
                return left.raw_score > right.raw_score;
            return left.chunk < right.chunk;
        });
        return output;
    }

    std::unordered_set<backend::ChunkKey> allowed_keys;
    if (!allowed.ids.empty())
        allowed_keys.insert(allowed.ids.begin(), allowed.ids.end());
    std::vector<bool> allowed_rows(keys_.size(), allowed.ids.empty());
    std::size_t allowed_count = allowed.ids.empty() ? keys_.size() : 0;
    if (!allowed.ids.empty())
        for (std::size_t row = 0; row < keys_.size(); ++row)
            if (allowed_keys.contains(keys_[row])) {
                allowed_rows[row] = true;
                ++allowed_count;
            }
    const std::size_t wanted = std::min(k, allowed_count);
    if (wanted == 0)
        return backend::CandidateList{};
    const auto allow = [&allowed_rows](std::uint32_t row) {
        return row < allowed_rows.size() && allowed_rows[row];
    };
    std::vector<Hit> hits;
    if (allowed_count == keys_.size()) {
        hits = graph_->search(query, wanted);
    } else {
        const float boost =
            std::max(4.0F, static_cast<float>(keys_.size()) / static_cast<float>(allowed_count));
        hits = graph_->search_filtered(query, wanted, allow, boost);
    }
    if (hits.size() < wanted)
        hits = graph_->search_exact_filtered(query, wanted, allow);

    backend::CandidateList output;
    output.reserve(hits.size());
    for (const auto& hit : hits) {
        const auto row = hit.chunk.get();
        if (row >= keys_.size() || !allowed_rows[row])
            return fail<backend::CandidateList>(Errc::corrupt_index,
                                                "native HNSW returned an invalid row");
        output.push_back({keys_[row], hit.score.get(), backend::ScoreType::cosine});
    }
    std::sort(output.begin(), output.end(), [](const auto& left, const auto& right) {
        if (left.raw_score != right.raw_score)
            return left.raw_score > right.raw_score;
        return left.chunk < right.chunk;
    });
    return output;
}

DenseIndexStats NativeHnswIndex::stats() const {
    std::shared_lock lock(mutex_);
    DenseIndexStats output;
    output.vectors = keys_.size();
    output.dimension = dimension_;
    output.exact = false;
    output.implementation = "native";
    output.algorithm = "hnsw";
    if (graph_) {
        const auto memory = graph_->memory_use();
        output.primary_vector_bytes = memory.vectors;
        output.compressed_vector_bytes = memory.sq8 + memory.pq;
        output.resident_bytes = memory.total();
    }
    for (const auto& key : keys_)
        output.resident_bytes += key.capacity();
    return output;
}

Result<std::string> NativeHnswIndex::serialize() const {
    try {
        std::unique_lock lock(mutex_);
        if (!graph_)
            return fail<std::string>(Errc::unavailable, "native HNSW is not built");
        return graph_->serialize();
    } catch (const std::exception& error) {
        return fail<std::string>(Errc::io_error,
                                 "native HNSW serialization failed: " + std::string(error.what()));
    } catch (...) {
        return fail<std::string>(Errc::io_error, "native HNSW serialization failed");
    }
}

} // namespace rag::dense
