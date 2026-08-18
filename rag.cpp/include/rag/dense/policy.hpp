#pragma once

#include <memory>
#include <string_view>

#include "rag/dense/index.hpp"
#include "rag/dense/native_hnsw.hpp"

namespace rag::dense {

enum class DenseImplementation { native, faiss };
enum class DenseAlgorithm { automatic, exact, hnsw, flat, ivf_sq8, ivf_pq };

struct DensePolicy {
    DenseImplementation implementation = DenseImplementation::native;
    DenseAlgorithm algorithm = DenseAlgorithm::automatic;
    std::size_t exact_threshold = 2'000;
    NativeHnswPolicy hnsw;
    struct FaissParameters {
        std::size_t hnsw_neighbors = 32;
        std::size_t ef_construction = 200;
        std::size_t ef_search = 64;
        std::size_t ivf_lists = 256;
        std::size_t ivf_probes = 16;
        std::size_t minimum_training_vectors_per_list = 39;
        std::size_t pq_subquantizers = 32;
        std::size_t pq_bits = 8;
    } faiss;
};

[[nodiscard]] constexpr std::string_view name(DenseImplementation implementation) noexcept {
    switch (implementation) {
        case DenseImplementation::native:
            return "native";
        case DenseImplementation::faiss:
            return "faiss";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view name(DenseAlgorithm algorithm) noexcept {
    switch (algorithm) {
        case DenseAlgorithm::automatic:
            return "automatic";
        case DenseAlgorithm::exact:
            return "exact";
        case DenseAlgorithm::hnsw:
            return "hnsw";
        case DenseAlgorithm::flat:
            return "flat";
        case DenseAlgorithm::ivf_sq8:
            return "ivf-sq8";
        case DenseAlgorithm::ivf_pq:
            return "ivf-pq";
    }
    return "unknown";
}

[[nodiscard]] Result<DenseImplementation> parse_dense_implementation(std::string_view value);
[[nodiscard]] Result<DenseAlgorithm> parse_dense_algorithm(std::string_view value);
[[nodiscard]] Result<DenseAlgorithm> resolve_dense_algorithm(const DensePolicy& policy,
                                                             std::size_t vector_count);
[[nodiscard]] Result<std::shared_ptr<DenseIndex>> build_dense_index(const DensePolicy& policy,
                                                                    VectorSource source);

} // namespace rag::dense
