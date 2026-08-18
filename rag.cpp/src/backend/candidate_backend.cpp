#include "rag/backend/candidate_backend.hpp"

#include <thread>

namespace rag::backend {

Result<CandidateBatch> CandidateBackend::hybrid_candidates(const HybridRequest& request) const {
    Result<CandidateList> dense = CandidateList{};
    std::thread helper([&] { dense = dense_candidates(request.dense); });
    auto lexical = lexical_candidates(request.lexical);
    helper.join();
    if (!lexical)
        return unexpected(lexical.error());
    if (!dense)
        return unexpected(dense.error());
    return CandidateBatch{std::move(*lexical), std::move(*dense)};
}

} // namespace rag::backend
