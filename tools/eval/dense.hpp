#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

#include <rag/core/types.hpp>
#include <rag/dense/policy.hpp>

namespace lrs::eval {

struct DenseOptions {
    std::size_t vectors = 25'000;
    std::size_t dimension = 384;
    std::size_t queries = 100;
    std::size_t k = 10;
    std::uint64_t seed = 0x6C72736576616C31ULL;
    rag::dense::DensePolicy policy;
    bool enforce_gate = false;
};

[[nodiscard]] rag::Result<nlohmann::json> run_dense(const DenseOptions& options);
[[nodiscard]] nlohmann::json corpus_manifest();

} // namespace lrs::eval
