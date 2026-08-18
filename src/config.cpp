#include "lrs/config.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace lrs {
Config load_config(const std::string& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open config: " + path);
    nlohmann::json j;
    input >> j;
    Config c;
    const auto& listen = j.value("listen", nlohmann::json::object());
    c.listen.host = listen.value("host", c.listen.host);
    c.listen.port = listen.value("port", c.listen.port);
    if (j.contains("index"))
        c.index_path = j.at("index").value("path", c.index_path);
    const auto& retrieval = j.value("retrieval", nlohmann::json::object());
    c.retrieval.backend = retrieval.value("backend", c.retrieval.backend);
    const auto& embedded = retrieval.value("embedded", nlohmann::json::object());
    c.retrieval.embedded.path = embedded.value("path", c.index_path);
    c.index_path = c.retrieval.embedded.path;
    const auto& dense = embedded.value("dense", nlohmann::json::object());
    c.retrieval.embedded.dense.implementation =
        dense.value("implementation", c.retrieval.embedded.dense.implementation);
    c.retrieval.embedded.dense.algorithm =
        dense.value("algorithm", c.retrieval.embedded.dense.algorithm);
    c.retrieval.embedded.dense.exact_threshold =
        dense.value("exact_threshold", c.retrieval.embedded.dense.exact_threshold);
    const auto& hnsw = dense.value("hnsw", nlohmann::json::object());
    auto& hnsw_config = c.retrieval.embedded.dense.hnsw;
    hnsw_config.neighbors = hnsw.value("neighbors", hnsw_config.neighbors);
    hnsw_config.ef_construction = hnsw.value("ef_construction", hnsw_config.ef_construction);
    hnsw_config.ef_search = hnsw.value("ef_search", hnsw_config.ef_search);
    hnsw_config.seed = hnsw.value("seed", hnsw_config.seed);
    const auto& faiss = dense.value("faiss", nlohmann::json::object());
    auto& faiss_config = c.retrieval.embedded.dense.faiss;
    faiss_config.hnsw_neighbors = faiss.value("hnsw_neighbors", faiss_config.hnsw_neighbors);
    faiss_config.ef_construction = faiss.value("ef_construction", faiss_config.ef_construction);
    faiss_config.ef_search = faiss.value("ef_search", faiss_config.ef_search);
    faiss_config.ivf_lists = faiss.value("ivf_lists", faiss_config.ivf_lists);
    faiss_config.ivf_probes = faiss.value("ivf_probes", faiss_config.ivf_probes);
    faiss_config.minimum_training_vectors_per_list = faiss.value(
        "minimum_training_vectors_per_list", faiss_config.minimum_training_vectors_per_list);
    faiss_config.pq_subquantizers = faiss.value("pq_subquantizers", faiss_config.pq_subquantizers);
    faiss_config.pq_bits = faiss.value("pq_bits", faiss_config.pq_bits);
    const auto& postgres = retrieval.value("postgres", nlohmann::json::object());
    auto& postgres_config = c.retrieval.postgres;
    postgres_config.connection_env =
        postgres.value("connection_env", postgres_config.connection_env);
    postgres_config.schema = postgres.value("schema", postgres_config.schema);
    postgres_config.corpus = postgres.value("corpus", postgres_config.corpus);
    postgres_config.pool_size = postgres.value("pool_size", postgres_config.pool_size);
    postgres_config.acquire_timeout_ms =
        postgres.value("acquire_timeout_ms", postgres_config.acquire_timeout_ms);
    postgres_config.statement_timeout_ms =
        postgres.value("statement_timeout_ms", postgres_config.statement_timeout_ms);
    postgres_config.vector_index = postgres.value("vector_index", postgres_config.vector_index);
    postgres_config.hnsw_ef_search =
        postgres.value("hnsw_ef_search", postgres_config.hnsw_ef_search);
    const auto& inference = j.at("inference");
    c.spawn = inference.value("spawn", c.spawn);
    c.llama_server = inference.value("llama_server", c.llama_server);
    const auto& embedding = inference.at("embedding");
    c.embedding.host = embedding.value("host", c.embedding.host);
    c.embedding.port = embedding.value("port", c.embedding.port);
    c.embedding_model = embedding.value("model", "");
    c.embedding_api_model = embedding.value("api_model", c.embedding_api_model);
    c.pooling = embedding.value("pooling", c.pooling);
    c.embedding_dimension = embedding.value("dimension", c.embedding_dimension);
    c.embedding_context_size = embedding.value("context_size", c.embedding_context_size);
    c.embedding_batch_size = embedding.value("batch_size", c.embedding_batch_size);
    c.deterministic_embeddings = embedding.value("deterministic", false);
    const auto& generation = inference.at("generation");
    c.generation.host = generation.value("host", c.generation.host);
    c.generation.port = generation.value("port", c.generation.port);
    c.generation_model = generation.value("model", "");
    c.generation_api_model = generation.value("api_model", c.generation_api_model);
    c.generation_context_size = generation.value("context_size", c.generation_context_size);
    c.context_tokens =
        j.value("rag", nlohmann::json::object()).value("context_tokens", c.context_tokens);
    const auto& ingestion = j.value("ingestion", nlohmann::json::object());
    c.ingestion_workers = ingestion.value("workers", c.ingestion_workers);
    c.ingestion_queue_capacity = ingestion.value("queue_capacity", c.ingestion_queue_capacity);
    const auto& authentication = j.value("authentication", nlohmann::json::object());
    c.api_key = authentication.value("api_key", c.api_key);
    validate_config(c);
    return c;
}

void validate_config(const Config& c) {
    const auto loopback = [](const std::string& host) {
        return host == "127.0.0.1" || host == "localhost" || host == "::1";
    };
    if (!loopback(c.embedding.host) || !loopback(c.generation.host))
        throw std::runtime_error("private model listeners must bind to loopback");
    if (!loopback(c.listen.host) && c.api_key.empty())
        throw std::runtime_error("a coordinator API key is required outside loopback");
    if (!c.api_key.empty()) {
        if (c.api_key.size() < 16)
            throw std::runtime_error("coordinator API key must contain at least 16 bytes");
        for (const unsigned char byte : c.api_key)
            if (byte < 0x21 || byte == 0x7f)
                throw std::runtime_error("coordinator API key contains invalid characters");
    }
    if (c.embedding.port == c.generation.port || c.listen.port == c.embedding.port ||
        c.listen.port == c.generation.port)
        throw std::runtime_error("listener ports must be distinct");
    if (c.pooling != "mean")
        throw std::runtime_error("embedding pooling must be mean");
    if (c.embedding_dimension == 0 || c.embedding_context_size == 0 ||
        c.embedding_batch_size < c.embedding_context_size || c.context_tokens == 0 ||
        c.context_tokens >= c.generation_context_size)
        throw std::runtime_error("invalid model dimension or context budget");
    if (c.ingestion_workers == 0 || c.ingestion_workers > 4 || c.ingestion_queue_capacity == 0)
        throw std::runtime_error("ingestion workers must be 1-4 and queue capacity non-zero");
    if (c.retrieval.backend != "embedded" && c.retrieval.backend != "postgres")
        throw std::runtime_error("configured retrieval backend is not available");
    if (c.retrieval.backend == "embedded" &&
        (c.index_path.empty() || c.retrieval.embedded.path.empty()))
        throw std::runtime_error("embedded retrieval path is invalid");
    const auto& dense = c.retrieval.embedded.dense;
    const bool native_algorithm =
        dense.algorithm == "automatic" || dense.algorithm == "exact" || dense.algorithm == "hnsw";
    const bool faiss_algorithm = dense.algorithm == "flat" || dense.algorithm == "hnsw" ||
                                 dense.algorithm == "ivf-sq8" || dense.algorithm == "ivf-pq";
    if (dense.exact_threshold == 0 || (dense.implementation == "native" && !native_algorithm) ||
        (dense.implementation == "faiss" && !faiss_algorithm) ||
        (dense.implementation != "native" && dense.implementation != "faiss"))
        throw std::runtime_error("invalid embedded dense implementation or algorithm");
#if !LRS_ENABLE_FAISS
    if (dense.implementation == "faiss")
        throw std::runtime_error("FAISS support is not enabled in this build");
#endif
    if (dense.hnsw.neighbors < 2 || dense.hnsw.neighbors > 128 ||
        dense.hnsw.ef_construction < dense.hnsw.neighbors || dense.hnsw.ef_search == 0)
        throw std::runtime_error("invalid native HNSW configuration");
    if (dense.faiss.hnsw_neighbors < 2 || dense.faiss.hnsw_neighbors > 128 ||
        dense.faiss.ef_construction < dense.faiss.hnsw_neighbors || dense.faiss.ef_search == 0 ||
        dense.faiss.ivf_lists == 0 || dense.faiss.ivf_probes == 0 ||
        dense.faiss.ivf_probes > dense.faiss.ivf_lists ||
        dense.faiss.minimum_training_vectors_per_list == 0 || dense.faiss.pq_subquantizers == 0 ||
        (dense.faiss.pq_bits != 8 && dense.faiss.pq_bits != 12 && dense.faiss.pq_bits != 16))
        throw std::runtime_error("invalid FAISS dense configuration");
    const auto identifier = [](const std::string& value) {
        if (value.empty() || value.size() > 63 ||
            !(std::isalpha(static_cast<unsigned char>(value.front())) || value.front() == '_'))
            return false;
        return std::all_of(value.begin() + 1, value.end(),
                           [](unsigned char byte) { return std::isalnum(byte) || byte == '_'; });
    };
    const auto& postgres = c.retrieval.postgres;
    if (c.retrieval.backend == "postgres") {
#if !LRS_ENABLE_POSTGRES
        throw std::runtime_error("PostgreSQL support is not enabled in this build");
#endif
        if (!identifier(postgres.connection_env) || !identifier(postgres.schema) ||
            postgres.corpus.empty() || postgres.pool_size == 0 || postgres.pool_size > 32 ||
            postgres.acquire_timeout_ms == 0 || postgres.statement_timeout_ms == 0 ||
            postgres.statement_timeout_ms > 600'000 ||
            (postgres.vector_index != "exact" && postgres.vector_index != "hnsw") ||
            postgres.hnsw_ef_search == 0 || postgres.hnsw_ef_search > 10'000)
            throw std::runtime_error("invalid PostgreSQL retrieval configuration");
    }
    if (c.spawn && (c.embedding_model.empty() || c.generation_model.empty()))
        throw std::runtime_error(
            "both GGUF model paths are required when process spawning is enabled");
}

bool authenticate_api_key(const Config& config, std::string_view supplied) noexcept {
    if (config.api_key.empty())
        return true;
    const std::string_view expected(config.api_key);
    const std::size_t width = std::max(expected.size(), supplied.size());
    std::size_t difference = expected.size() ^ supplied.size();
    for (std::size_t i = 0; i < width; ++i) {
        const unsigned char left =
            i < expected.size() ? static_cast<unsigned char>(expected[i]) : 0;
        const unsigned char right =
            i < supplied.size() ? static_cast<unsigned char>(supplied[i]) : 0;
        difference |= static_cast<std::size_t>(left ^ right);
    }
    return difference == 0;
}

bool authorize_api_request(const Config& config, std::size_t header_count,
                           std::string_view supplied) noexcept {
    return config.api_key.empty() || (header_count == 1 && authenticate_api_key(config, supplied));
}
} // namespace lrs
