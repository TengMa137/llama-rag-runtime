#include "rag/dense/hnsw_sidecar.hpp"

#include <exception>
#include <new>

#include "rag/dense/cache_fingerprint.hpp"
#include "rag/store/container.hpp"
#include "rag/store/container_view.hpp"

namespace rag::dense {
namespace {

constexpr std::uint32_t kSidecarVersion = 1;

std::string parameters(const NativeHnswPolicy& policy) {
    return "M=" + std::to_string(policy.neighbors) +
           ";efc=" + std::to_string(policy.ef_construction) +
           ";efs=" + std::to_string(policy.ef_search) + ";seed=" + std::to_string(policy.seed);
}

} // namespace

std::string hnsw_sidecar_path(std::string_view checkpoint_path) {
    return std::string(checkpoint_path) + ".dense.native-hnsw-v1";
}

std::string hnsw_sidecar_fingerprint(VectorSource source, std::string_view embedding_identity,
                                     const NativeHnswPolicy& policy) {
    return dense_cache_fingerprint(source, embedding_identity,
                                   {"native", "hnsw", kSidecarVersion, parameters(policy)});
}

Result<void> write_hnsw_sidecar(const std::string& path, std::string_view fingerprint,
                                const NativeHnswIndex& index) {
    try {
        if (fingerprint.empty())
            return fail<void>(Errc::invalid_argument, "HNSW sidecar fingerprint is required");
        const auto stats = index.stats();
        if (stats.algorithm != "hnsw" || stats.compressed_vector_bytes != 0 ||
            stats.vectors != index.keys().size())
            return fail<void>(Errc::invalid_argument, "HNSW sidecar index is incompatible");
        auto graph = index.serialize();
        if (!graph)
            return unexpected(graph.error());

        store::Writer metadata;
        metadata.u<std::uint32_t>(kSidecarVersion);
        metadata.str("native");
        metadata.str("hnsw");
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
        container.put(store::Tag::hnsw, std::move(*graph));
        return container.write_file(path);
    } catch (const std::exception& error) {
        return fail<void>(Errc::io_error,
                          "HNSW sidecar write failed: " + std::string(error.what()));
    } catch (...) {
        return fail<void>(Errc::io_error, "HNSW sidecar write failed");
    }
}

Result<std::shared_ptr<NativeHnswIndex>>
load_hnsw_sidecar(const std::string& path, std::string_view expected_fingerprint,
                  std::span<const backend::ChunkKey> expected_keys,
                  const NativeHnswPolicy& policy) {
    try {
        auto container = store::ContainerView::open_file(path);
        if (!container)
            return unexpected(container.error());
        const auto metadata_blob = container->get(store::Tag::meta);
        const auto keys_blob = container->get(store::Tag::chunks);
        const auto graph_blob = container->get(store::Tag::hnsw);
        if (!metadata_blob || !keys_blob || !graph_blob)
            return fail<std::shared_ptr<NativeHnswIndex>>(Errc::corrupt_index,
                                                          "HNSW sidecar sections are invalid");

        store::Reader metadata(*metadata_blob);
        std::uint32_t version = 0;
        std::string implementation;
        std::string algorithm;
        std::string fingerprint;
        std::uint64_t rows = 0;
        std::uint64_t dimension = 0;
        if (!metadata.u(version) || version != kSidecarVersion || !metadata.str(implementation) ||
            implementation != "native" || !metadata.str(algorithm) || algorithm != "hnsw" ||
            !metadata.str(fingerprint) || fingerprint != expected_fingerprint ||
            !metadata.u(rows) || !metadata.u(dimension) || metadata.remaining() != 0 ||
            rows != expected_keys.size())
            return fail<std::shared_ptr<NativeHnswIndex>>(Errc::corrupt_index,
                                                          "HNSW sidecar metadata is invalid");

        store::Reader key_reader(*keys_blob);
        std::uint64_t key_count = 0;
        if (!key_reader.u(key_count) || key_count != expected_keys.size())
            return fail<std::shared_ptr<NativeHnswIndex>>(Errc::corrupt_index,
                                                          "HNSW sidecar key count is invalid");
        std::vector<backend::ChunkKey> keys;
        keys.reserve(expected_keys.size());
        for (std::size_t index = 0; index < expected_keys.size(); ++index) {
            backend::ChunkKey key;
            if (!key_reader.str(key) || key != expected_keys[index])
                return fail<std::shared_ptr<NativeHnswIndex>>(
                    Errc::corrupt_index, "HNSW sidecar key catalog is invalid");
            keys.push_back(std::move(key));
        }
        if (key_reader.remaining() != 0)
            return fail<std::shared_ptr<NativeHnswIndex>>(Errc::corrupt_index,
                                                          "HNSW sidecar has trailing keys");
        auto loaded = NativeHnswIndex::from_serialized(*graph_blob, keys, policy);
        if (!loaded || (*loaded)->stats().dimension != dimension)
            return fail<std::shared_ptr<NativeHnswIndex>>(Errc::corrupt_index,
                                                          "HNSW sidecar graph is invalid");
        return loaded;
    } catch (const std::bad_alloc&) {
        return fail<std::shared_ptr<NativeHnswIndex>>(Errc::corrupt_index,
                                                      "HNSW sidecar allocation failed");
    } catch (const std::exception& error) {
        return fail<std::shared_ptr<NativeHnswIndex>>(
            Errc::corrupt_index, "HNSW sidecar load failed: " + std::string(error.what()));
    } catch (...) {
        return fail<std::shared_ptr<NativeHnswIndex>>(Errc::corrupt_index,
                                                      "HNSW sidecar load failed");
    }
}

} // namespace rag::dense
