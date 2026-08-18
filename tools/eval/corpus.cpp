#include "corpus.hpp"

#include <iomanip>
#include <sstream>

#include <rag/dense/simd.hpp>

namespace lrs::eval {

std::vector<rag::dense::VectorRecord> CorpusData::records() const {
    std::vector<rag::dense::VectorRecord> output;
    output.reserve(vectors.size());
    for (std::size_t row = 0; row < vectors.size(); ++row)
        output.push_back({keys[row], vectors[row]});
    return output;
}

std::uint64_t random_u64(std::uint64_t& state) {
    state ^= state >> 12U;
    state ^= state << 25U;
    state ^= state >> 27U;
    return state * 2685821657736338717ULL;
}

float random_unit(std::uint64_t& state) {
    const std::uint32_t bits = static_cast<std::uint32_t>(random_u64(state) >> 40U);
    return static_cast<float>(bits) / static_cast<float>(0xFFFFFFU) * 2.0F - 1.0F;
}

CorpusData generate_corpus(std::size_t count, std::size_t dimension, std::uint64_t seed) {
    CorpusData output;
    output.vectors.reserve(count);
    output.keys.reserve(count);
    const std::size_t width = count == 0 ? 1 : std::to_string(count - 1).size();
    for (std::size_t row = 0; row < count; ++row) {
        rag::Vector vector(dimension);
        for (float& value : vector)
            value = random_unit(seed);
        rag::dense::normalize(vector);
        output.vectors.push_back(std::move(vector));
        std::ostringstream key;
        key << "chk_eval_" << std::setfill('0') << std::setw(static_cast<int>(width)) << row;
        output.keys.push_back(key.str());
    }
    return output;
}

rag::Vector perturbed_query(rag::VectorView source, std::uint64_t& state) {
    rag::Vector query(source.begin(), source.end());
    for (float& value : query)
        value += random_unit(state) * 0.0025F;
    rag::dense::normalize(query);
    return query;
}

} // namespace lrs::eval
