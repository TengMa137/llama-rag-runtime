#pragma once

#include <vector>

#include "rag/backend/candidate_backend.hpp"
#include "rag/core/document.hpp"

namespace rag::retrieval {

struct RuntimeConfig {
    float bm25_weight = 1.0F;
    float dense_weight = 1.0F;
    float mmr_lambda = 0.5F;
    std::size_t rrf_rank_constant = 60;
    std::size_t adjacent_line_gap = 1;
};

// Backend-independent candidate fusion, feature reranking, optional MMR,
// stitching, and resolution. Dense/hybrid requests provide a normalized query
// embedding prepared by the caller's embedding policy.
[[nodiscard]] Result<std::vector<SearchResult>> search(const backend::CandidateBackend& backend,
                                                       const backend::SearchRequest& request,
                                                       RuntimeConfig config = {});

} // namespace rag::retrieval
