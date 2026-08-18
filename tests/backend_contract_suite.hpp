#pragma once

#include <cstddef>
#include <functional>
#include <string>

#include <rag/backend/candidate_backend.hpp>
#include <rag/core/types.hpp>

namespace lrs::tests {

struct CandidateBackendContractOptions {
    std::string key_prefix;
    std::size_t dimension = 16;
    std::size_t filler_chunks = 0;
    bool require_durable = false;
    std::function<rag::Result<void>()> publish_base;
};

// Mutates an otherwise empty backend through the complete public candidate
// contract. Callers own backend construction, persistence, and cleanup.
[[nodiscard]] rag::Result<void>
run_candidate_backend_contract(rag::backend::CandidateBackend& backend,
                               const CandidateBackendContractOptions& options);

} // namespace lrs::tests
