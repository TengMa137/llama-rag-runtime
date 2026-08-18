#include "rag/dense/faiss_index.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <unordered_set>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

namespace rag::dense {
namespace {

bool normalized(VectorView vector) {
    if (vector.empty())
        return false;
    double norm = 0.0;
    for (const float value : vector) {
        if (!std::isfinite(value))
            return false;
        norm += static_cast<double>(value) * value;
    }
    return std::abs(norm - 1.0) <= 1.0e-3;
}

Result<void> validate_parameters(DenseAlgorithm algorithm,
                                 const DensePolicy::FaissParameters& parameters, std::size_t rows,
                                 std::size_t dimension) {
    if (algorithm != DenseAlgorithm::flat && algorithm != DenseAlgorithm::hnsw &&
        algorithm != DenseAlgorithm::ivf_sq8 && algorithm != DenseAlgorithm::ivf_pq)
        return fail<void>(Errc::invalid_argument, "unsupported FAISS algorithm");
    if (algorithm == DenseAlgorithm::hnsw &&
        (parameters.hnsw_neighbors < 2 || parameters.hnsw_neighbors > 128 ||
         parameters.ef_construction < parameters.hnsw_neighbors || parameters.ef_search == 0))
        return fail<void>(Errc::invalid_argument, "invalid FAISS HNSW parameters");
    if (algorithm == DenseAlgorithm::ivf_sq8 || algorithm == DenseAlgorithm::ivf_pq) {
        if (parameters.ivf_lists == 0 || parameters.ivf_probes == 0 ||
            parameters.ivf_probes > parameters.ivf_lists ||
            parameters.minimum_training_vectors_per_list == 0 ||
            parameters.ivf_lists > std::numeric_limits<std::size_t>::max() /
                                       parameters.minimum_training_vectors_per_list ||
            rows < parameters.ivf_lists * parameters.minimum_training_vectors_per_list)
            return fail<void>(Errc::invalid_argument,
                              "FAISS IVF training corpus is too small for configured lists");
    }
    if (algorithm == DenseAlgorithm::ivf_pq &&
        (parameters.pq_subquantizers == 0 || dimension % parameters.pq_subquantizers != 0 ||
         (parameters.pq_bits != 8 && parameters.pq_bits != 12 && parameters.pq_bits != 16)))
        return fail<void>(Errc::invalid_argument, "invalid FAISS IVF-PQ parameters");
    if (algorithm == DenseAlgorithm::ivf_pq) {
        const std::size_t pq_centroids = std::size_t{1} << parameters.pq_bits;
        if (pq_centroids > std::numeric_limits<std::size_t>::max() /
                               parameters.minimum_training_vectors_per_list ||
            rows < pq_centroids * parameters.minimum_training_vectors_per_list)
            return fail<void>(Errc::invalid_argument,
                              "FAISS IVF-PQ training corpus is too small for its codebooks");
    }
    return {};
}

std::size_t estimate_bytes(const faiss::Index& index, DenseAlgorithm algorithm,
                           const DensePolicy::FaissParameters& parameters) {
    const auto rows = static_cast<std::size_t>(index.ntotal);
    const auto dimension = static_cast<std::size_t>(index.d);
    switch (algorithm) {
        case DenseAlgorithm::flat:
            return rows * dimension * sizeof(float);
        case DenseAlgorithm::hnsw:
            return rows * dimension * sizeof(float) +
                   rows * parameters.hnsw_neighbors * 2 * sizeof(faiss::idx_t);
        case DenseAlgorithm::ivf_sq8:
            return rows * (dimension + sizeof(faiss::idx_t)) +
                   parameters.ivf_lists * dimension * sizeof(float);
        case DenseAlgorithm::ivf_pq:
            return rows * ((parameters.pq_subquantizers * parameters.pq_bits + 7) / 8 +
                           sizeof(faiss::idx_t)) +
                   parameters.ivf_lists * dimension * sizeof(float);
        default:
            return 0;
    }
}

bool compatible_index(const faiss::Index& index, DenseAlgorithm algorithm,
                      const DensePolicy::FaissParameters& parameters) {
    if (index.metric_type != faiss::METRIC_INNER_PRODUCT || !index.is_trained)
        return false;
    if (algorithm == DenseAlgorithm::flat)
        return dynamic_cast<const faiss::IndexFlatIP*>(&index) != nullptr;
    if (algorithm == DenseAlgorithm::hnsw) {
        const auto* hnsw = dynamic_cast<const faiss::IndexHNSWFlat*>(&index);
        return hnsw != nullptr &&
               hnsw->hnsw.nb_neighbors(0) == static_cast<int>(parameters.hnsw_neighbors * 2) &&
               hnsw->hnsw.efConstruction == static_cast<int>(parameters.ef_construction);
    }
    if (algorithm == DenseAlgorithm::ivf_sq8) {
        const auto* scalar = dynamic_cast<const faiss::IndexIVFScalarQuantizer*>(&index);
        return scalar != nullptr && scalar->nlist == parameters.ivf_lists &&
               scalar->sq.qtype == faiss::ScalarQuantizer::QT_8bit;
    }
    if (algorithm == DenseAlgorithm::ivf_pq) {
        const auto* pq = dynamic_cast<const faiss::IndexIVFPQ*>(&index);
        return pq != nullptr && pq->nlist == parameters.ivf_lists &&
               pq->pq.M == parameters.pq_subquantizers && pq->pq.nbits == parameters.pq_bits;
    }
    return false;
}

} // namespace

struct FaissIndex::Impl {
    // IndexIVF borrows its coarse quantizer by default. Declaration order makes
    // the index die first and the quantizer second.
    std::unique_ptr<faiss::Index> quantizer;
    std::unique_ptr<faiss::Index> index;
    std::size_t resident_bytes = 0;
};

FaissIndex::FaissIndex(DenseAlgorithm algorithm, DensePolicy::FaissParameters parameters)
    : algorithm_(algorithm), parameters_(parameters) {}

FaissIndex::~FaissIndex() = default;

Result<std::shared_ptr<FaissIndex>>
FaissIndex::from_serialized(std::string_view blob, std::span<const backend::ChunkKey> keys,
                            DenseAlgorithm algorithm, DensePolicy::FaissParameters parameters) {
    try {
        faiss::VectorIOReader reader;
        reader.data.assign(blob.begin(), blob.end());
        auto restored = faiss::read_index_up(&reader);
        if (!restored || reader.rp != reader.data.size() || restored->ntotal < 0 ||
            static_cast<std::size_t>(restored->ntotal) != keys.size() ||
            !compatible_index(*restored, algorithm, parameters))
            return fail<std::shared_ptr<FaissIndex>>(Errc::corrupt_index,
                                                     "FAISS cache index is incompatible");
        std::unordered_set<backend::ChunkKey> unique;
        for (const auto& key : keys)
            if (key.empty() || !unique.insert(key).second)
                return fail<std::shared_ptr<FaissIndex>>(Errc::corrupt_index,
                                                         "FAISS cache keys are invalid");
        auto output = std::make_shared<FaissIndex>(algorithm, parameters);
        output->impl_ = std::make_unique<Impl>();
        output->impl_->resident_bytes = estimate_bytes(*restored, algorithm, parameters);
        output->impl_->index = std::move(restored);
        output->keys_.assign(keys.begin(), keys.end());
        output->dimension_ = static_cast<std::size_t>(output->impl_->index->d);
        return output;
    } catch (const std::exception& error) {
        return fail<std::shared_ptr<FaissIndex>>(
            Errc::corrupt_index, "FAISS cache load failed: " + std::string(error.what()));
    } catch (...) {
        return fail<std::shared_ptr<FaissIndex>>(Errc::corrupt_index, "FAISS cache load failed");
    }
}

Result<void> FaissIndex::build(VectorSource source) {
    try {
        if (source.size() > static_cast<std::size_t>(std::numeric_limits<faiss::idx_t>::max()))
            return fail<void>(Errc::invalid_argument, "FAISS row limit exceeded");
        const std::size_t dimension = source.empty() ? 0 : source.front().vector.size();
        if (dimension > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            return fail<void>(Errc::invalid_argument, "FAISS dimension limit exceeded");
        if (auto valid = validate_parameters(algorithm_, parameters_, source.size(), dimension);
            !valid)
            return valid;

        std::vector<backend::ChunkKey> keys;
        std::unordered_set<backend::ChunkKey> unique;
        std::vector<float> matrix;
        keys.reserve(source.size());
        if (dimension != 0 && source.size() > matrix.max_size() / dimension)
            return fail<void>(Errc::invalid_argument, "FAISS matrix size overflow");
        matrix.reserve(source.size() * dimension);
        for (const auto& row : source) {
            if (row.key.empty() || !unique.emplace(row.key).second)
                return fail<void>(Errc::invalid_argument, "FAISS keys are invalid");
            if (row.vector.size() != dimension)
                return fail<void>(Errc::dimension_mismatch, "FAISS vector dimensions differ");
            if (!normalized(row.vector))
                return fail<void>(Errc::invalid_argument, "FAISS vector is not normalized");
            keys.emplace_back(row.key);
            matrix.insert(matrix.end(), row.vector.begin(), row.vector.end());
        }

        auto built = std::make_unique<Impl>();
        if (source.empty()) {
            std::unique_lock lock(mutex_);
            impl_ = std::move(built);
            keys_.clear();
            dimension_ = 0;
            return {};
        }
        const auto faiss_dimension = static_cast<faiss::idx_t>(dimension);
        switch (algorithm_) {
            case DenseAlgorithm::flat:
                built->index = std::make_unique<faiss::IndexFlatIP>(faiss_dimension);
                built->resident_bytes = matrix.size() * sizeof(float);
                break;
            case DenseAlgorithm::hnsw: {
                auto hnsw = std::make_unique<faiss::IndexHNSWFlat>(
                    static_cast<int>(dimension), static_cast<int>(parameters_.hnsw_neighbors),
                    faiss::METRIC_INNER_PRODUCT);
                hnsw->hnsw.efConstruction = static_cast<int>(parameters_.ef_construction);
                hnsw->hnsw.efSearch = static_cast<int>(parameters_.ef_search);
                built->resident_bytes =
                    matrix.size() * sizeof(float) +
                    source.size() * parameters_.hnsw_neighbors * 2 * sizeof(faiss::idx_t);
                built->index = std::move(hnsw);
                break;
            }
            case DenseAlgorithm::ivf_sq8:
                built->quantizer = std::make_unique<faiss::IndexFlatIP>(faiss_dimension);
                built->index = std::make_unique<faiss::IndexIVFScalarQuantizer>(
                    built->quantizer.get(), faiss_dimension, parameters_.ivf_lists,
                    faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_INNER_PRODUCT);
                built->resident_bytes = source.size() * (dimension + sizeof(faiss::idx_t)) +
                                        parameters_.ivf_lists * dimension * sizeof(float);
                break;
            case DenseAlgorithm::ivf_pq:
                built->quantizer = std::make_unique<faiss::IndexFlatIP>(faiss_dimension);
                built->index = std::make_unique<faiss::IndexIVFPQ>(
                    built->quantizer.get(), faiss_dimension, parameters_.ivf_lists,
                    parameters_.pq_subquantizers, parameters_.pq_bits, faiss::METRIC_INNER_PRODUCT);
                built->resident_bytes =
                    source.size() * ((parameters_.pq_subquantizers * parameters_.pq_bits + 7) / 8 +
                                     sizeof(faiss::idx_t)) +
                    parameters_.ivf_lists * dimension * sizeof(float);
                break;
            default:
                return fail<void>(Errc::invalid_argument, "unsupported FAISS algorithm");
        }
        if (!source.empty()) {
            if (!built->index->is_trained)
                built->index->train(static_cast<faiss::idx_t>(source.size()), matrix.data());
            built->index->add(static_cast<faiss::idx_t>(source.size()), matrix.data());
        }

        std::unique_lock lock(mutex_);
        impl_ = std::move(built);
        keys_ = std::move(keys);
        dimension_ = dimension;
        return {};
    } catch (const std::exception& error) {
        return fail<void>(Errc::invalid_argument,
                          "FAISS build failed: " + std::string(error.what()));
    } catch (...) {
        return fail<void>(Errc::invalid_argument, "FAISS build failed");
    }
}

Result<backend::CandidateList> FaissIndex::search(Vector query, AllowedIds allowed,
                                                  std::size_t k) const {
    try {
        std::shared_lock lock(mutex_);
        if (k == 0 || keys_.empty())
            return backend::CandidateList{};
        if (!impl_ || !impl_->index)
            return fail<backend::CandidateList>(Errc::unavailable, "FAISS index is not built");
        if (query.size() != dimension_)
            return fail<backend::CandidateList>(Errc::dimension_mismatch,
                                                "query dimension does not match FAISS index");
        if (!normalized(query))
            return fail<backend::CandidateList>(Errc::invalid_argument,
                                                "FAISS query is not normalized");

        std::unordered_set<backend::ChunkKey> allowed_keys;
        if (!allowed.ids.empty())
            allowed_keys.insert(allowed.ids.begin(), allowed.ids.end());
        if (!allowed_keys.empty() &&
            (algorithm_ == DenseAlgorithm::flat || algorithm_ == DenseAlgorithm::hnsw)) {
            backend::CandidateList exact;
            Vector row(dimension_);
            for (std::size_t index = 0; index < keys_.size(); ++index) {
                if (!allowed_keys.contains(keys_[index]))
                    continue;
                impl_->index->reconstruct(static_cast<faiss::idx_t>(index), row.data());
                float score = 0.0F;
                for (std::size_t component = 0; component < dimension_; ++component)
                    score += query[component] * row[component];
                exact.push_back({keys_[index], score, backend::ScoreType::cosine});
            }
            const auto order = [](const auto& left, const auto& right) {
                if (left.raw_score != right.raw_score)
                    return left.raw_score > right.raw_score;
                return left.chunk < right.chunk;
            };
            const auto wanted = std::min(k, exact.size());
            std::partial_sort(exact.begin(), exact.begin() + static_cast<std::ptrdiff_t>(wanted),
                              exact.end(), order);
            exact.resize(wanted);
            return exact;
        }

        std::vector<faiss::idx_t> allowed_labels;
        if (!allowed_keys.empty()) {
            allowed_labels.reserve(allowed_keys.size());
            for (std::size_t index = 0; index < keys_.size(); ++index)
                if (allowed_keys.contains(keys_[index]))
                    allowed_labels.push_back(static_cast<faiss::idx_t>(index));
        }
        const std::size_t raw_k =
            algorithm_ == DenseAlgorithm::flat
                ? keys_.size()
                : std::min(k, allowed_keys.empty() ? keys_.size() : allowed_labels.size());
        if (raw_k == 0)
            return backend::CandidateList{};
        std::vector<float> distances(raw_k);
        std::vector<faiss::idx_t> labels(raw_k);
        faiss::SearchParametersHNSW hnsw_parameters;
        faiss::SearchParametersIVF ivf_parameters;
        std::unique_ptr<faiss::IDSelectorBatch> selector;
        const faiss::SearchParameters* search_parameters = nullptr;
        if (algorithm_ == DenseAlgorithm::hnsw) {
            hnsw_parameters.efSearch =
                static_cast<int>(allowed.ids.empty() ? parameters_.ef_search : keys_.size());
            search_parameters = &hnsw_parameters;
        } else if (algorithm_ == DenseAlgorithm::ivf_sq8 || algorithm_ == DenseAlgorithm::ivf_pq) {
            ivf_parameters.nprobe = static_cast<faiss::idx_t>(
                allowed.ids.empty() ? parameters_.ivf_probes : parameters_.ivf_lists);
            if (!allowed_labels.empty()) {
                selector = std::make_unique<faiss::IDSelectorBatch>(allowed_labels.size(),
                                                                    allowed_labels.data());
                ivf_parameters.sel = selector.get();
            }
            search_parameters = &ivf_parameters;
        }
        impl_->index->search(1, query.data(), static_cast<faiss::idx_t>(raw_k), distances.data(),
                             labels.data(), search_parameters);

        backend::CandidateList output;
        output.reserve(std::min(k, raw_k));
        for (std::size_t rank = 0; rank < raw_k; ++rank) {
            const auto label = labels[rank];
            if (label < 0 || static_cast<std::size_t>(label) >= keys_.size())
                continue;
            const auto& key = keys_[static_cast<std::size_t>(label)];
            if (!allowed_keys.empty() && !allowed_keys.contains(key))
                continue;
            output.push_back({key, distances[rank], backend::ScoreType::cosine});
            if (output.size() == k)
                break;
        }
        const std::size_t available = allowed.ids.empty() ? keys_.size() : allowed_labels.size();
        if (output.size() != std::min(k, available))
            return fail<backend::CandidateList>(Errc::corrupt_index,
                                                "FAISS filtered search returned incomplete top-k");
        std::sort(output.begin(), output.end(), [](const auto& left, const auto& right) {
            if (left.raw_score != right.raw_score)
                return left.raw_score > right.raw_score;
            return left.chunk < right.chunk;
        });
        return output;
    } catch (const std::exception& error) {
        return fail<backend::CandidateList>(Errc::unavailable,
                                            "FAISS search failed: " + std::string(error.what()));
    } catch (...) {
        return fail<backend::CandidateList>(Errc::unavailable, "FAISS search failed");
    }
}

DenseIndexStats FaissIndex::stats() const {
    std::shared_lock lock(mutex_);
    DenseIndexStats output;
    output.vectors = keys_.size();
    output.dimension = dimension_;
    output.exact = algorithm_ == DenseAlgorithm::flat;
    output.implementation = "faiss";
    output.algorithm = std::string(name(algorithm_));
    if (impl_)
        output.resident_bytes = impl_->resident_bytes;
    if (algorithm_ == DenseAlgorithm::flat || algorithm_ == DenseAlgorithm::hnsw)
        output.primary_vector_bytes = keys_.size() * dimension_ * sizeof(float);
    else
        output.compressed_vector_bytes = output.resident_bytes;
    for (const auto& key : keys_)
        output.resident_bytes += key.capacity();
    return output;
}

Result<std::string> FaissIndex::serialize() const {
    try {
        std::unique_lock lock(mutex_);
        if (!impl_ || !impl_->index)
            return fail<std::string>(Errc::unavailable, "FAISS index is not built");
        faiss::VectorIOWriter writer;
        faiss::write_index(impl_->index.get(), &writer);
        return std::string(reinterpret_cast<const char*>(writer.data.data()), writer.data.size());
    } catch (const std::exception& error) {
        return fail<std::string>(Errc::io_error,
                                 "FAISS cache serialization failed: " + std::string(error.what()));
    } catch (...) {
        return fail<std::string>(Errc::io_error, "FAISS cache serialization failed");
    }
}

} // namespace rag::dense
