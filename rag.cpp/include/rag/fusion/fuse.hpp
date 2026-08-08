#pragma once
// Deterministic weighted Reciprocal Rank Fusion.

#include <span>
#include <vector>

#include "rag/core/types.hpp"

namespace rag::fusion {
struct RankedList {
    std::vector<Hit> hits;
    float weight = 1.0F;
};

[[nodiscard]] inline RankedList bm25_list(std::vector<Hit> hits, float weight = 1.0F) {
    return RankedList{std::move(hits), weight};
}
[[nodiscard]] inline RankedList cosine_list(std::vector<Hit> hits, float weight = 1.0F) {
    return RankedList{std::move(hits), weight};
}

struct RrfParams {
    float k = 60.0F;
};

[[nodiscard]] std::vector<Hit> rrf(std::span<const RankedList> lists, RrfParams params = {},
                                   std::size_t top_k = 0);
} // namespace rag::fusion
