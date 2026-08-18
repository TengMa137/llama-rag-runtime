#pragma once

#include <cstddef>
#include <cstdint>

#include <nlohmann/json.hpp>

#include <rag/core/types.hpp>
#include <rag/dense/policy.hpp>

namespace lrs::eval {

struct RuntimeOptions {
    std::size_t base_vectors = 25'000;
    std::size_t dimension = 384;
    std::size_t queries = 100;
    std::size_t k = 10;
    std::uint64_t seed = 0x6C72736576616C31ULL;
    rag::dense::DensePolicy policy;
};

[[nodiscard]] rag::Result<nlohmann::json> run_runtime(const RuntimeOptions& options);

} // namespace lrs::eval
