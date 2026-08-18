#include "rag/dense/policy.hpp"

#if LRS_ENABLE_FAISS
#include "rag/dense/faiss_index.hpp"
#endif

namespace rag::dense {

Result<DenseImplementation> parse_dense_implementation(std::string_view value) {
    if (value == "native")
        return DenseImplementation::native;
    if (value == "faiss")
        return DenseImplementation::faiss;
    return fail<DenseImplementation>(Errc::invalid_argument,
                                     "unknown dense implementation: " + std::string(value));
}

Result<DenseAlgorithm> parse_dense_algorithm(std::string_view value) {
    if (value == "automatic")
        return DenseAlgorithm::automatic;
    if (value == "exact")
        return DenseAlgorithm::exact;
    if (value == "hnsw")
        return DenseAlgorithm::hnsw;
    if (value == "flat")
        return DenseAlgorithm::flat;
    if (value == "ivf-sq8")
        return DenseAlgorithm::ivf_sq8;
    if (value == "ivf-pq")
        return DenseAlgorithm::ivf_pq;
    return fail<DenseAlgorithm>(Errc::invalid_argument,
                                "unknown dense algorithm: " + std::string(value));
}

Result<DenseAlgorithm> resolve_dense_algorithm(const DensePolicy& policy,
                                               std::size_t vector_count) {
    if (policy.exact_threshold == 0)
        return fail<DenseAlgorithm>(Errc::invalid_argument,
                                    "dense exact threshold must be non-zero");
    if (policy.implementation == DenseImplementation::native) {
        if (policy.algorithm == DenseAlgorithm::automatic)
            return vector_count < policy.exact_threshold ? DenseAlgorithm::exact
                                                         : DenseAlgorithm::hnsw;
        if (policy.algorithm == DenseAlgorithm::exact || policy.algorithm == DenseAlgorithm::hnsw)
            return policy.algorithm;
        return fail<DenseAlgorithm>(Errc::invalid_argument,
                                    "dense algorithm is not supported by native implementation");
    }
    if (policy.algorithm == DenseAlgorithm::flat || policy.algorithm == DenseAlgorithm::hnsw ||
        policy.algorithm == DenseAlgorithm::ivf_sq8 || policy.algorithm == DenseAlgorithm::ivf_pq) {
#if LRS_ENABLE_FAISS
        return policy.algorithm;
#else
        return fail<DenseAlgorithm>(Errc::unavailable,
                                    "FAISS support is not enabled in this build");
#endif
    }
    return fail<DenseAlgorithm>(Errc::invalid_argument,
                                "dense algorithm is not supported by FAISS implementation");
}

Result<std::shared_ptr<DenseIndex>> build_dense_index(const DensePolicy& policy,
                                                      VectorSource source) {
    auto algorithm = resolve_dense_algorithm(policy, source.size());
    if (!algorithm)
        return unexpected(algorithm.error());
    std::shared_ptr<DenseIndex> index;
    switch (*algorithm) {
        case DenseAlgorithm::exact:
            index = std::make_shared<NativeExactIndex>();
            break;
        case DenseAlgorithm::hnsw:
            if (policy.implementation == DenseImplementation::native)
                index = std::make_shared<NativeHnswIndex>(policy.hnsw);
#if LRS_ENABLE_FAISS
            else
                index = std::make_shared<FaissIndex>(*algorithm, policy.faiss);
#endif
            break;
        case DenseAlgorithm::flat:
        case DenseAlgorithm::ivf_sq8:
        case DenseAlgorithm::ivf_pq:
#if LRS_ENABLE_FAISS
            index = std::make_shared<FaissIndex>(*algorithm, policy.faiss);
            break;
#else
            return fail<std::shared_ptr<DenseIndex>>(Errc::unavailable,
                                                     "FAISS support is not enabled in this build");
#endif
        default:
            return fail<std::shared_ptr<DenseIndex>>(Errc::invalid_argument,
                                                     "resolved dense algorithm is unsupported");
    }
    if (auto built = index->build(source); !built)
        return unexpected(built.error());
    return index;
}

} // namespace rag::dense
