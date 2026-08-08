#include "rag/fusion/fuse.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace rag::fusion {
std::vector<Hit> rrf(std::span<const RankedList> lists, RrfParams params, std::size_t top_k) {
    if (!std::isfinite(params.k) || params.k < 0.0F)
        return {};
    std::unordered_map<std::uint32_t, float> accumulated;
    for (const auto& list : lists) {
        if (!std::isfinite(list.weight) || list.weight < 0.0F)
            continue;
        for (std::size_t rank = 0; rank < list.hits.size(); ++rank) {
            const std::uint32_t id = list.hits[rank].chunk.get();
            accumulated[id] += list.weight / (params.k + static_cast<float>(rank) + 1.0F);
        }
    }
    std::vector<Hit> output;
    output.reserve(accumulated.size());
    for (const auto& [id, score] : accumulated)
        output.push_back(Hit{ChunkId{id}, Score{score}});
    std::sort(output.begin(), output.end(), [](const Hit& left, const Hit& right) {
        if (left.score.get() != right.score.get())
            return left.score.get() > right.score.get();
        return left.chunk.get() < right.chunk.get();
    });
    if (top_k != 0 && output.size() > top_k)
        output.resize(top_k);
    return output;
}
} // namespace rag::fusion
