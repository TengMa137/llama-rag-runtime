#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <rag/backend/candidate_backend.hpp>
#include <rag/dense/index.hpp>

namespace lrs::eval {

struct CorpusData {
    std::vector<rag::Vector> vectors;
    std::vector<rag::backend::ChunkKey> keys;
    [[nodiscard]] std::vector<rag::dense::VectorRecord> records() const;
};

[[nodiscard]] std::uint64_t random_u64(std::uint64_t& state);
[[nodiscard]] float random_unit(std::uint64_t& state);
[[nodiscard]] CorpusData generate_corpus(std::size_t vectors, std::size_t dimension,
                                         std::uint64_t seed);
[[nodiscard]] rag::Vector perturbed_query(rag::VectorView source, std::uint64_t& state);

} // namespace lrs::eval
