#include "rag/dense/faiss_sidecar.hpp"

#include <exception>
#include <new>

#include <faiss/Index.h>

#include "rag/dense/cache_fingerprint.hpp"
#include "rag/store/container.hpp"
#include "rag/store/container_view.hpp"

namespace rag::dense {
namespace {

constexpr std::uint32_t kSidecarVersion = 1;
constexpr std::uint32_t kFaissIndexTag = 0x58444944; // "DIDX"

std::string parameters_string(const DensePolicy::FaissParameters& parameters) {
    return std::string("faiss=" VERSION_STRING ";M=") + std::to_string(parameters.hnsw_neighbors) +
           ";efc=" + std::to_string(parameters.ef_construction) +
           ";efs=" + std::to_string(parameters.ef_search) +
           ";nlist=" + std::to_string(parameters.ivf_lists) +
           ";nprobe=" + std::to_string(parameters.ivf_probes) +
           ";mintrain=" + std::to_string(parameters.minimum_training_vectors_per_list) +
           ";pq-m=" + std::to_string(parameters.pq_subquantizers) +
           ";pq-bits=" + std::to_string(parameters.pq_bits);
}

bool supported(DenseAlgorithm algorithm) {
    return algorithm == DenseAlgorithm::flat || algorithm == DenseAlgorithm::hnsw ||
           algorithm == DenseAlgorithm::ivf_sq8 || algorithm == DenseAlgorithm::ivf_pq;
}

} // namespace

std::string faiss_sidecar_path(std::string_view checkpoint_path, DenseAlgorithm algorithm) {
    return std::string(checkpoint_path) + ".dense.faiss-" + std::string(name(algorithm)) + "-v1";
}

std::string faiss_sidecar_fingerprint(VectorSource source, std::string_view embedding_identity,
                                      DenseAlgorithm algorithm,
                                      const DensePolicy::FaissParameters& parameters) {
    return dense_cache_fingerprint(
        source, embedding_identity,
        {"faiss", name(algorithm), kSidecarVersion, parameters_string(parameters)});
}

Result<void> write_faiss_sidecar(const std::string& path, std::string_view fingerprint,
                                 const FaissIndex& index) {
    try {
        if (fingerprint.empty())
            return fail<void>(Errc::invalid_argument, "FAISS sidecar fingerprint is required");
        const auto stats = index.stats();
        if (stats.implementation != "faiss" || stats.vectors != index.keys().size())
            return fail<void>(Errc::invalid_argument, "FAISS sidecar index is incompatible");
        const auto serialized = index.serialize();
        if (!serialized)
            return unexpected(serialized.error());

        store::Writer metadata;
        metadata.u<std::uint32_t>(kSidecarVersion);
        metadata.str("faiss");
        metadata.str(stats.algorithm);
        metadata.str(fingerprint);
        metadata.u<std::uint64_t>(stats.vectors);
        metadata.u<std::uint64_t>(stats.dimension);
        store::Writer keys;
        keys.u<std::uint64_t>(index.keys().size());
        for (const auto& key : index.keys())
            keys.str(key);

        store::Container container;
        container.put(store::Tag::meta, std::move(metadata.data()));
        container.put(store::Tag::chunks, std::move(keys.data()));
        container.put_raw(kFaissIndexTag, std::move(*serialized));
        return container.write_file(path);
    } catch (const std::exception& error) {
        return fail<void>(Errc::io_error,
                          "FAISS sidecar write failed: " + std::string(error.what()));
    } catch (...) {
        return fail<void>(Errc::io_error, "FAISS sidecar write failed");
    }
}

Result<std::shared_ptr<FaissIndex>>
load_faiss_sidecar(const std::string& path, std::string_view expected_fingerprint,
                   std::span<const backend::ChunkKey> expected_keys, DenseAlgorithm algorithm,
                   const DensePolicy::FaissParameters& parameters) {
    try {
        if (!supported(algorithm))
            return fail<std::shared_ptr<FaissIndex>>(Errc::invalid_argument,
                                                     "unsupported FAISS sidecar algorithm");
        auto container = store::ContainerView::open_file(path);
        if (!container)
            return unexpected(container.error());
        const auto metadata_blob = container->get(store::Tag::meta);
        const auto keys_blob = container->get(store::Tag::chunks);
        const auto index_blob = container->get_raw(kFaissIndexTag);
        if (!metadata_blob || !keys_blob || !index_blob)
            return fail<std::shared_ptr<FaissIndex>>(Errc::corrupt_index,
                                                     "FAISS sidecar sections are invalid");

        store::Reader metadata(*metadata_blob);
        std::uint32_t version = 0;
        std::string implementation;
        std::string stored_algorithm;
        std::string fingerprint;
        std::uint64_t rows = 0;
        std::uint64_t dimension = 0;
        if (!metadata.u(version) || version != kSidecarVersion || !metadata.str(implementation) ||
            implementation != "faiss" || !metadata.str(stored_algorithm) ||
            stored_algorithm != name(algorithm) || !metadata.str(fingerprint) ||
            fingerprint != expected_fingerprint || !metadata.u(rows) || !metadata.u(dimension) ||
            metadata.remaining() != 0 || rows != expected_keys.size())
            return fail<std::shared_ptr<FaissIndex>>(Errc::corrupt_index,
                                                     "FAISS sidecar metadata is invalid");

        store::Reader key_reader(*keys_blob);
        std::uint64_t key_count = 0;
        if (!key_reader.u(key_count) || key_count != expected_keys.size())
            return fail<std::shared_ptr<FaissIndex>>(Errc::corrupt_index,
                                                     "FAISS sidecar key count is invalid");
        std::vector<backend::ChunkKey> keys;
        keys.reserve(expected_keys.size());
        for (std::size_t index = 0; index < expected_keys.size(); ++index) {
            backend::ChunkKey key;
            if (!key_reader.str(key) || key != expected_keys[index])
                return fail<std::shared_ptr<FaissIndex>>(Errc::corrupt_index,
                                                         "FAISS sidecar key catalog is invalid");
            keys.push_back(std::move(key));
        }
        if (key_reader.remaining() != 0)
            return fail<std::shared_ptr<FaissIndex>>(Errc::corrupt_index,
                                                     "FAISS sidecar has trailing keys");
        auto loaded = FaissIndex::from_serialized(*index_blob, keys, algorithm, parameters);
        if (!loaded || (*loaded)->stats().dimension != dimension)
            return fail<std::shared_ptr<FaissIndex>>(Errc::corrupt_index,
                                                     "FAISS sidecar index is invalid");
        return loaded;
    } catch (const std::bad_alloc&) {
        return fail<std::shared_ptr<FaissIndex>>(Errc::corrupt_index,
                                                 "FAISS sidecar allocation failed");
    } catch (const std::exception& error) {
        return fail<std::shared_ptr<FaissIndex>>(
            Errc::corrupt_index, "FAISS sidecar load failed: " + std::string(error.what()));
    } catch (...) {
        return fail<std::shared_ptr<FaissIndex>>(Errc::corrupt_index, "FAISS sidecar load failed");
    }
}

} // namespace rag::dense
