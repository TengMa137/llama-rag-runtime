#include "rag/backend/postgres_backend.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <string_view>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "../postgres/connection_pool.hpp"
#include "../postgres/migrations.hpp"
#include "rag/core/keys.hpp"
#include "rag/text/tokenizer.hpp"

namespace rag::backend {
namespace {

using postgres::Connection;
using postgres::ConnectionPool;
using postgres::QueryResult;

struct CorpusState {
    std::uint64_t id = 0;
    std::size_t dimension = 0;
    std::string embedding_identity;
    std::string chunking_fingerprint;
    std::uint64_t generation = 0;
};

template <class Integer> Result<Integer> integer(std::string_view value, std::string_view field) {
    Integer output{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), output);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        return fail<Integer>(Errc::corrupt_index,
                             "PostgreSQL returned an invalid " + std::string(field));
    return output;
}

Result<float> real(std::string_view value, std::string_view field) {
    try {
        std::size_t parsed = 0;
        const float output = std::stof(std::string(value), &parsed);
        if (parsed != value.size() || !std::isfinite(output))
            return fail<float>(Errc::corrupt_index,
                               "PostgreSQL returned an invalid " + std::string(field));
        return output;
    } catch (...) {
        return fail<float>(Errc::corrupt_index,
                           "PostgreSQL returned an invalid " + std::string(field));
    }
}

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

std::string vector_text(VectorView vector) {
    std::ostringstream output;
    output << '[' << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t index = 0; index < vector.size(); ++index) {
        if (index != 0)
            output << ',';
        output << vector[index];
    }
    output << ']';
    return output.str();
}

Result<Vector> parse_vector(std::string_view value) {
    if (value.size() < 2 || value.front() != '[' || value.back() != ']')
        return fail<Vector>(Errc::corrupt_index, "PostgreSQL returned an invalid vector");
    value.remove_prefix(1);
    value.remove_suffix(1);
    Vector output;
    while (!value.empty()) {
        const auto separator = value.find(',');
        const auto item = value.substr(0, separator);
        auto parsed = real(item, "vector component");
        if (!parsed)
            return unexpected(parsed.error());
        output.push_back(*parsed);
        if (separator == std::string_view::npos)
            break;
        value.remove_prefix(separator + 1);
    }
    if (!normalized(output))
        return fail<Vector>(Errc::corrupt_index, "PostgreSQL returned a non-normalized vector");
    return output;
}

std::string metadata_json(const Metadata& metadata) { return nlohmann::json(metadata).dump(); }

std::string string_array_json(std::span<const std::string> values) {
    return nlohmann::json(values).dump();
}

Result<void> validate_document(const PreparedDocument& document, DocumentRevision revision) {
    if (document.key.empty() || revision == 0 || document.content_hash.empty() ||
        document.chunking_fingerprint.empty() || document.embedding_identity.empty() ||
        document.chunks.empty())
        return fail<void>(Errc::invalid_argument, "prepared document identity is incomplete");
    if (document.content != normalize_source_text(document.content) ||
        document.content_hash !=
            document_content_hash(document.title, document.content, document.metadata))
        return fail<void>(Errc::invalid_argument, "prepared document content hash is invalid");
    std::size_t dimension = 0;
    std::unordered_map<ChunkKey, bool> unique;
    for (std::size_t index = 0; index < document.chunks.size(); ++index) {
        const auto& chunk = document.chunks[index];
        const std::string indexed =
            chunk.context.empty() ? chunk.text : chunk.context + "\n" + chunk.text;
        if (chunk.key.empty() || !unique.emplace(chunk.key, true).second ||
            chunk.ordinal != index || chunk.text.empty() || chunk.indexed_text.empty() ||
            chunk.start_line > chunk.end_line || chunk.indexed_text != indexed ||
            chunk.key !=
                stable_chunk_key(document.key, chunk.start_line, chunk.end_line, chunk.text))
            return fail<void>(Errc::invalid_argument, "prepared chunk contract is invalid");
        if (!normalized(chunk.embedding))
            return fail<void>(Errc::invalid_argument, "prepared chunk embedding is invalid");
        if (dimension == 0)
            dimension = chunk.embedding.size();
        if (chunk.embedding.size() != dimension)
            return fail<void>(Errc::dimension_mismatch, "prepared chunk dimensions differ");
    }
    return {};
}

Result<std::optional<CorpusState>> corpus_state(Connection& connection, std::string_view corpus) {
    const std::array<std::string, 1> parameters{std::string(corpus)};
    auto result = connection.execute_prepared("lrs_corpus_get", parameters);
    if (!result)
        return unexpected(result.error());
    if (result->rows() == 0)
        return std::optional<CorpusState>{};
    if (result->rows() != 1 || result->columns() != 5)
        return fail<std::optional<CorpusState>>(Errc::corrupt_index,
                                                "PostgreSQL corpus catalog is invalid");
    auto id = integer<std::uint64_t>(result->value(0, 0), "corpus ID");
    auto dimension = integer<std::size_t>(result->value(0, 1), "corpus dimension");
    auto generation = integer<std::uint64_t>(result->value(0, 4), "corpus generation");
    if (!id || !dimension || !generation)
        return fail<std::optional<CorpusState>>(Errc::corrupt_index,
                                                "PostgreSQL corpus catalog is invalid");
    return std::optional<CorpusState>{CorpusState{*id, *dimension, std::string(result->value(0, 2)),
                                                  std::string(result->value(0, 3)), *generation}};
}

Result<void> prepare_connection(Connection& connection, void* opaque) {
    const auto& config = *static_cast<const PostgresConfig*>(opaque);
    const auto q = postgres::qualified_schema(config.schema);
    struct Statement {
        const char* name;
        std::string sql;
        std::size_t parameters;
    };
    const std::vector<Statement> statements{
        {"lrs_corpus_get",
         "SELECT id, dimension, embedding_identity, chunking_fingerprint, generation FROM " + q +
             ".corpora WHERE name=$1",
         1},
        {"lrs_corpus_create",
         "INSERT INTO " + q +
             ".corpora(name,dimension,metric,embedding_identity,chunking_fingerprint) "
             "VALUES($1,$2::integer,'cosine',$3,$4) ON CONFLICT(name) DO NOTHING",
         4},
        {"lrs_revision_lock",
         "SELECT revision,deleted,content_hash FROM " + q +
             ".document_revisions WHERE corpus_id=$1::bigint AND external_id=$2 FOR UPDATE",
         2},
        {"lrs_revision_put",
         "INSERT INTO " + q +
             ".document_revisions(corpus_id,external_id,revision,deleted,content_hash) "
             "VALUES($1::bigint,$2,$3::bigint,$4::boolean,$5) "
             "ON CONFLICT(corpus_id,external_id) DO UPDATE SET revision=EXCLUDED.revision,"
             "deleted=EXCLUDED.deleted,content_hash=EXCLUDED.content_hash",
         5},
        {"lrs_document_delete",
         "DELETE FROM " + q + ".documents WHERE corpus_id=$1::bigint AND external_id=$2", 2},
        {"lrs_document_insert",
         "INSERT INTO " + q +
             ".documents(corpus_id,external_id,revision,title,content,metadata,content_hash) "
             "VALUES($1::bigint,$2,$3::bigint,$4,$5,$6::jsonb,$7)",
         7},
        {"lrs_chunk_insert",
         "INSERT INTO " + q +
             ".chunks(corpus_id,document_id,document_revision,chunk_key,ordinal,text,indexed_text,"
             "context,start_line,end_line,token_count,embedding) VALUES("
             "$1::bigint,$2,$3::bigint,$4,$5::integer,$6,$7,$8,$9::integer,$10::integer,"
             "$11::integer,$12::vector)",
         12},
        {"lrs_term_insert",
         "INSERT INTO " + q + ".terms(corpus_id,term) VALUES($1::bigint,$2) ON CONFLICT DO NOTHING",
         2},
        {"lrs_posting_insert",
         "INSERT INTO " + q +
             ".postings(corpus_id,term,chunk_key,term_frequency) "
             "VALUES($1::bigint,$2,$3,$4::integer)",
         4},
        {"lrs_generation_increment",
         "UPDATE " + q + ".corpora SET generation=generation+1 WHERE id=$1::bigint", 1},
        {"lrs_lexical",
         "WITH query_terms AS ("
         " SELECT value AS term,count(*)::real AS qtf FROM "
         " jsonb_array_elements_text($3::jsonb) GROUP BY value),"
         " corpus_stats AS (SELECT count(*)::real AS n,"
         " COALESCE(avg(token_count),1)::real AS avgdl FROM " +
             q +
             ".chunks WHERE corpus_id=$1::bigint),"
             " dfs AS (SELECT p.term,count(*)::real AS df FROM " +
             q +
             ".postings p JOIN query_terms qt ON qt.term=p.term "
             " WHERE p.corpus_id=$1::bigint GROUP BY p.term) "
             "SELECT p.chunk_key,(sum(qt.qtf * ln(1 + (cs.n-df.df+0.5)/(df.df+0.5)) * "
             "(p.term_frequency::real*2.2)/(p.term_frequency::real + 1.2*(0.25 + "
             "0.75*c.token_count::real/GREATEST(cs.avgdl,1)))))::real AS score FROM " +
             q +
             ".postings p JOIN query_terms qt ON qt.term=p.term JOIN dfs df ON df.term=p.term "
             "JOIN " +
             q + ".chunks c ON c.corpus_id=p.corpus_id AND c.chunk_key=p.chunk_key JOIN " + q +
             ".documents d ON d.corpus_id=c.corpus_id AND d.external_id=c.document_id "
             "CROSS JOIN corpus_stats cs WHERE p.corpus_id=$1::bigint AND d.metadata @> $2::jsonb "
             "GROUP BY p.chunk_key ORDER BY score DESC,p.chunk_key ASC LIMIT $4::bigint",
         4},
        {"lrs_dense_exact",
         "SELECT c.chunk_key,-(c.embedding <#> $3::vector)::real AS score FROM " + q +
             ".chunks c JOIN " + q +
             ".documents d ON d.corpus_id=c.corpus_id AND d.external_id=c.document_id "
             "WHERE c.corpus_id=$1::bigint AND d.metadata @> $2::jsonb "
             "ORDER BY c.embedding <#> $3::vector,c.chunk_key ASC LIMIT $4::bigint",
         4},
        {"lrs_fetch",
         "WITH wanted AS (SELECT value AS chunk_key,ordinality FROM "
         "jsonb_array_elements_text($2::jsonb) WITH ORDINALITY) "
         "SELECT c.chunk_key,c.document_id,c.document_revision,c.ordinal,d.title,c.text,"
         "c.context,d.metadata::text,c.start_line,c.end_line,"
         "CASE WHEN $3::boolean THEN c.embedding::text ELSE NULL END FROM wanted w JOIN " +
             q + ".chunks c ON c.corpus_id=$1::bigint AND c.chunk_key=w.chunk_key JOIN " + q +
             ".documents d ON d.corpus_id=c.corpus_id AND d.external_id=c.document_id "
             "ORDER BY w.ordinality",
         3},
        {"lrs_stats",
         "SELECT (SELECT count(*) FROM " + q +
             ".documents d WHERE d.corpus_id=cp.id),"
             "(SELECT count(*) FROM " +
             q + ".chunks c WHERE c.corpus_id=cp.id),cp.dimension,cp.generation FROM " + q +
             ".corpora cp WHERE cp.name=$1",
         1}};
    for (const auto& statement : statements)
        if (auto prepared = connection.prepare(statement.name, statement.sql, statement.parameters);
            !prepared)
            return prepared;
    if (config.vector_index == PostgresVectorIndex::hnsw) {
        if (auto configured = connection.execute("SET hnsw.iterative_scan = strict_order");
            !configured)
            return unexpected(configured.error());
        if (auto configured =
                connection.execute("SET hnsw.ef_search = " + std::to_string(config.hnsw_ef_search));
            !configured)
            return unexpected(configured.error());
    }
    return {};
}

Result<void> ensure_hnsw_index(Connection& connection, const PostgresConfig& config,
                               const CorpusState& corpus) {
    if (config.vector_index != PostgresVectorIndex::hnsw)
        return {};
    const auto q = postgres::qualified_schema(config.schema);
    const std::string name = "lrs_chunks_hnsw_" + std::to_string(corpus.id);
    const std::string sql = "CREATE INDEX IF NOT EXISTS \"" + name + "\" ON " + q +
                            ".chunks USING hnsw ((embedding::vector(" +
                            std::to_string(corpus.dimension) +
                            ")) vector_ip_ops) WHERE corpus_id=" + std::to_string(corpus.id);
    auto created = connection.execute(sql);
    if (!created)
        return unexpected(created.error());
    return {};
}

Result<QueryResult> hnsw_candidates(Connection& connection, const PostgresConfig& config,
                                    const CorpusState& corpus, const DenseRequest& request) {
    const auto q = postgres::qualified_schema(config.schema);
    const std::string name = "lrs_dense_hnsw_" + std::to_string(corpus.id);
    const std::string dimension = std::to_string(corpus.dimension);
    const std::string sql =
        "SELECT c.chunk_key,-((c.embedding::vector(" + dimension + ")) <#> $2::vector(" +
        dimension + "))::real AS score FROM " + q + ".chunks c JOIN " + q +
        ".documents d ON d.corpus_id=c.corpus_id AND d.external_id=c.document_id "
        "WHERE c.corpus_id=" +
        std::to_string(corpus.id) + " AND d.metadata @> $1::jsonb ORDER BY (c.embedding::vector(" +
        dimension + ")) <#> $2::vector(" + dimension + "),c.chunk_key ASC LIMIT $3::bigint";
    if (auto prepared = connection.prepare(name, sql, 3); !prepared)
        return unexpected(prepared.error());
    const std::array<std::string, 3> parameters{metadata_json(request.filter.required),
                                                vector_text(request.query),
                                                std::to_string(request.k)};
    return connection.execute_prepared(name, parameters);
}

Result<void> rollback(Connection& connection, Error error) {
    (void)connection.execute("ROLLBACK");
    return unexpected(std::move(error));
}

} // namespace

struct PostgresBackend::Impl {
    PostgresConfig config;
    std::shared_ptr<ConnectionPool> pool;

    [[nodiscard]] Result<CandidateList> lexical(Connection& connection,
                                                const LexicalRequest& request) const {
        if (request.k == 0)
            return CandidateList{};
        auto corpus = corpus_state(connection, config.corpus);
        if (!corpus)
            return unexpected(corpus.error());
        if (!*corpus)
            return CandidateList{};
        text::Tokenizer tokenizer;
        const auto terms = tokenizer.tokenize(request.query);
        if (terms.empty())
            return CandidateList{};
        const std::array<std::string, 4> parameters{
            std::to_string((*corpus)->id), metadata_json(request.filter.required),
            string_array_json(terms), std::to_string(request.k)};
        auto result = connection.execute_prepared("lrs_lexical", parameters);
        if (!result)
            return unexpected(result.error());
        CandidateList output;
        output.reserve(result->rows());
        for (std::size_t row = 0; row < result->rows(); ++row) {
            auto score = real(result->value(row, 1), "BM25 score");
            if (!score)
                return unexpected(score.error());
            output.push_back({std::string(result->value(row, 0)), *score, ScoreType::bm25});
        }
        return output;
    }

    [[nodiscard]] Result<CandidateList> dense(Connection& connection,
                                              const DenseRequest& request) const {
        if (request.k == 0)
            return CandidateList{};
        if (!normalized(request.query))
            return fail<CandidateList>(Errc::invalid_argument,
                                       "PostgreSQL query vector is not normalized");
        auto corpus = corpus_state(connection, config.corpus);
        if (!corpus)
            return unexpected(corpus.error());
        if (!*corpus)
            return CandidateList{};
        if (request.query.size() != (*corpus)->dimension)
            return fail<CandidateList>(Errc::dimension_mismatch,
                                       "query dimension does not match PostgreSQL corpus");
        Result<QueryResult> result =
            fail<QueryResult>(Errc::unavailable, "dense query was not initialized");
        if (config.vector_index == PostgresVectorIndex::hnsw) {
            result = hnsw_candidates(connection, config, **corpus, request);
        } else {
            const std::array<std::string, 4> parameters{
                std::to_string((*corpus)->id), metadata_json(request.filter.required),
                vector_text(request.query), std::to_string(request.k)};
            result = connection.execute_prepared("lrs_dense_exact", parameters);
        }
        if (!result)
            return unexpected(result.error());
        CandidateList output;
        output.reserve(result->rows());
        for (std::size_t row = 0; row < result->rows(); ++row) {
            auto score = real(result->value(row, 1), "dense score");
            if (!score)
                return unexpected(score.error());
            output.push_back({std::string(result->value(row, 0)), *score, ScoreType::cosine});
        }
        return output;
    }
};

PostgresBackend::PostgresBackend(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

PostgresBackend::~PostgresBackend() {
    if (impl_ && impl_->pool)
        impl_->pool->cancel_all();
}

Result<std::shared_ptr<PostgresBackend>> PostgresBackend::open(PostgresConfig config) {
    if (auto valid = validate_postgres_config(config); !valid)
        return unexpected(valid.error());
    auto bootstrap = Connection::open(config);
    if (!bootstrap)
        return unexpected(bootstrap.error());
    if (config.run_migrations) {
        if (auto migrated = postgres::migrate(*bootstrap, config.schema); !migrated)
            return unexpected(migrated.error());
    }
    auto pool = ConnectionPool::open(config, prepare_connection, &config);
    if (!pool)
        return unexpected(pool.error());
    auto impl = std::make_unique<Impl>();
    impl->config = std::move(config);
    impl->pool = std::move(*pool);
    auto output = std::shared_ptr<PostgresBackend>(new PostgresBackend(std::move(impl)));
    if (output->impl_->config.vector_index == PostgresVectorIndex::hnsw) {
        auto lease = output->impl_->pool->acquire(output->impl_->config.acquire_timeout);
        if (!lease)
            return unexpected(lease.error());
        auto corpus = corpus_state(lease->connection(), output->impl_->config.corpus);
        if (!corpus)
            return unexpected(corpus.error());
        if (*corpus)
            if (auto indexed =
                    ensure_hnsw_index(lease->connection(), output->impl_->config, **corpus);
                !indexed)
                return unexpected(indexed.error());
    }
    return output;
}

Result<void> PostgresBackend::activate(PreparedDocument document, DocumentRevision revision) {
    if (auto valid = validate_document(document, revision); !valid)
        return valid;
    auto lease = impl_->pool->acquire(impl_->config.acquire_timeout);
    if (!lease)
        return unexpected(lease.error());
    auto& connection = lease->connection();
    if (auto begun = connection.execute("BEGIN ISOLATION LEVEL SERIALIZABLE"); !begun)
        return unexpected(begun.error());

    const auto dimension = document.chunks.front().embedding.size();
    const std::array<std::string, 4> create_parameters{
        impl_->config.corpus, std::to_string(dimension), document.embedding_identity,
        document.chunking_fingerprint};
    if (auto created = connection.execute_prepared("lrs_corpus_create", create_parameters);
        !created)
        return rollback(connection, created.error());
    auto corpus = corpus_state(connection, impl_->config.corpus);
    if (!corpus || !*corpus)
        return rollback(connection, corpus ? Error{Errc::corrupt_index, "corpus creation failed"}
                                           : corpus.error());
    if ((*corpus)->dimension != dimension ||
        (*corpus)->embedding_identity != document.embedding_identity ||
        (*corpus)->chunking_fingerprint != document.chunking_fingerprint)
        return rollback(connection,
                        Error{Errc::dimension_mismatch,
                              "prepared document is incompatible with PostgreSQL corpus"});

    const std::array<std::string, 2> identity{std::to_string((*corpus)->id), document.key};
    auto current = connection.execute_prepared("lrs_revision_lock", identity);
    if (!current)
        return rollback(connection, current.error());
    if (current->rows() != 0) {
        auto stored_revision = integer<DocumentRevision>(current->value(0, 0), "document revision");
        if (!stored_revision)
            return rollback(connection, stored_revision.error());
        if (revision <= *stored_revision) {
            const bool same = revision == *stored_revision && current->value(0, 1) == "f" &&
                              current->value(0, 2) == document.content_hash;
            (void)connection.execute(same ? "COMMIT" : "ROLLBACK");
            return same ? Result<void>{}
                        : fail<void>(Errc::already_exists, "document revision is stale");
        }
    }

    if (auto removed = connection.execute_prepared("lrs_document_delete", identity); !removed)
        return rollback(connection, removed.error());
    const std::array<std::string, 7> document_parameters{
        identity[0],          document.key,     std::to_string(revision),
        document.title,       document.content, metadata_json(document.metadata),
        document.content_hash};
    if (auto inserted = connection.execute_prepared("lrs_document_insert", document_parameters);
        !inserted)
        return rollback(connection, inserted.error());

    text::Tokenizer tokenizer;
    for (const auto& chunk : document.chunks) {
        const auto terms = tokenizer.tokenize(chunk.indexed_text);
        std::map<std::string, std::size_t> frequencies;
        for (const auto& term : terms)
            ++frequencies[term];
        const std::array<std::string, 12> chunk_parameters{identity[0],
                                                           document.key,
                                                           std::to_string(revision),
                                                           chunk.key,
                                                           std::to_string(chunk.ordinal),
                                                           chunk.text,
                                                           chunk.indexed_text,
                                                           chunk.context,
                                                           std::to_string(chunk.start_line),
                                                           std::to_string(chunk.end_line),
                                                           std::to_string(terms.size()),
                                                           vector_text(chunk.embedding)};
        if (auto inserted = connection.execute_prepared("lrs_chunk_insert", chunk_parameters);
            !inserted)
            return rollback(connection, inserted.error());
        for (const auto& [term, frequency] : frequencies) {
            const std::array<std::string, 2> term_parameters{identity[0], term};
            if (auto inserted = connection.execute_prepared("lrs_term_insert", term_parameters);
                !inserted)
                return rollback(connection, inserted.error());
            const std::array<std::string, 4> posting_parameters{identity[0], term, chunk.key,
                                                                std::to_string(frequency)};
            if (auto inserted =
                    connection.execute_prepared("lrs_posting_insert", posting_parameters);
                !inserted)
                return rollback(connection, inserted.error());
        }
    }
    const std::array<std::string, 5> revision_parameters{
        identity[0], document.key, std::to_string(revision), "false", document.content_hash};
    if (auto saved = connection.execute_prepared("lrs_revision_put", revision_parameters); !saved)
        return rollback(connection, saved.error());
    const std::array<std::string, 1> corpus_id{identity[0]};
    if (auto updated = connection.execute_prepared("lrs_generation_increment", corpus_id); !updated)
        return rollback(connection, updated.error());
    if (auto indexed = ensure_hnsw_index(connection, impl_->config, **corpus); !indexed)
        return rollback(connection, indexed.error());
    auto committed = connection.execute("COMMIT");
    if (!committed)
        return rollback(connection, committed.error());
    return {};
}

Result<bool> PostgresBackend::erase(DocumentKey document, DocumentRevision revision) {
    if (document.empty() || revision == 0)
        return fail<bool>(Errc::invalid_argument, "document key and revision are required");
    auto lease = impl_->pool->acquire(impl_->config.acquire_timeout);
    if (!lease)
        return unexpected(lease.error());
    auto& connection = lease->connection();
    if (auto begun = connection.execute("BEGIN ISOLATION LEVEL SERIALIZABLE"); !begun)
        return unexpected(begun.error());
    auto corpus = corpus_state(connection, impl_->config.corpus);
    if (!corpus)
        return unexpected(rollback(connection, corpus.error()).error());
    if (!*corpus) {
        (void)connection.execute("COMMIT");
        return false;
    }
    const std::array<std::string, 2> identity{std::to_string((*corpus)->id), document};
    auto current = connection.execute_prepared("lrs_revision_lock", identity);
    if (!current)
        return unexpected(rollback(connection, current.error()).error());
    if (current->rows() != 0) {
        auto stored_revision = integer<DocumentRevision>(current->value(0, 0), "document revision");
        if (!stored_revision)
            return unexpected(rollback(connection, stored_revision.error()).error());
        if (revision <= *stored_revision) {
            const bool same_delete = revision == *stored_revision && current->value(0, 1) == "t";
            (void)connection.execute(same_delete ? "COMMIT" : "ROLLBACK");
            if (same_delete)
                return false;
            return fail<bool>(Errc::already_exists, "document revision is stale");
        }
    }
    auto removed = connection.execute_prepared("lrs_document_delete", identity);
    if (!removed)
        return unexpected(rollback(connection, removed.error()).error());
    const bool existed = removed->affected_rows() != 0;
    const std::array<std::string, 5> revision_parameters{
        identity[0], document, std::to_string(revision), "true", "deleted"};
    if (auto saved = connection.execute_prepared("lrs_revision_put", revision_parameters); !saved)
        return unexpected(rollback(connection, saved.error()).error());
    const std::array<std::string, 1> corpus_id{identity[0]};
    if (auto updated = connection.execute_prepared("lrs_generation_increment", corpus_id); !updated)
        return unexpected(rollback(connection, updated.error()).error());
    if (auto committed = connection.execute("COMMIT"); !committed)
        return unexpected(rollback(connection, committed.error()).error());
    return existed;
}

Result<CandidateList> PostgresBackend::lexical_candidates(const LexicalRequest& request) const {
    auto lease = impl_->pool->acquire(impl_->config.acquire_timeout);
    if (!lease)
        return unexpected(lease.error());
    return impl_->lexical(lease->connection(), request);
}

Result<CandidateList> PostgresBackend::dense_candidates(const DenseRequest& request) const {
    auto lease = impl_->pool->acquire(impl_->config.acquire_timeout);
    if (!lease)
        return unexpected(lease.error());
    return impl_->dense(lease->connection(), request);
}

Result<CandidateBatch> PostgresBackend::hybrid_candidates(const HybridRequest& request) const {
    auto lease = impl_->pool->acquire(impl_->config.acquire_timeout);
    if (!lease)
        return unexpected(lease.error());
    auto& connection = lease->connection();
    if (auto begun = connection.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY"); !begun)
        return unexpected(begun.error());
    auto lexical = impl_->lexical(connection, request.lexical);
    if (!lexical)
        return unexpected(rollback(connection, lexical.error()).error());
    auto dense = impl_->dense(connection, request.dense);
    if (!dense)
        return unexpected(rollback(connection, dense.error()).error());
    if (auto committed = connection.execute("COMMIT"); !committed)
        return unexpected(rollback(connection, committed.error()).error());
    return CandidateBatch{std::move(*lexical), std::move(*dense)};
}

Result<std::vector<StoredChunk>> PostgresBackend::fetch(std::span<const ChunkKey> chunks,
                                                        FetchOptions options) const {
    if (chunks.empty())
        return std::vector<StoredChunk>{};
    auto lease = impl_->pool->acquire(impl_->config.acquire_timeout);
    if (!lease)
        return unexpected(lease.error());
    auto corpus = corpus_state(lease->connection(), impl_->config.corpus);
    if (!corpus)
        return unexpected(corpus.error());
    if (!*corpus)
        return std::vector<StoredChunk>{};
    const std::array<std::string, 3> parameters{std::to_string((*corpus)->id),
                                                string_array_json(chunks),
                                                options.include_embedding ? "true" : "false"};
    auto result = lease->connection().execute_prepared("lrs_fetch", parameters);
    if (!result)
        return unexpected(result.error());
    std::vector<StoredChunk> output;
    output.reserve(result->rows());
    for (std::size_t row = 0; row < result->rows(); ++row) {
        auto revision = integer<DocumentRevision>(result->value(row, 2), "chunk revision");
        auto ordinal = integer<std::size_t>(result->value(row, 3), "chunk ordinal");
        auto start_line = integer<std::uint32_t>(result->value(row, 8), "chunk start line");
        auto end_line = integer<std::uint32_t>(result->value(row, 9), "chunk end line");
        if (!revision || !ordinal || !start_line || !end_line)
            return fail<std::vector<StoredChunk>>(Errc::corrupt_index,
                                                  "PostgreSQL chunk row is invalid");
        StoredChunk chunk;
        chunk.key = result->value(row, 0);
        chunk.document = result->value(row, 1);
        chunk.revision = *revision;
        chunk.ordinal = *ordinal;
        chunk.title = result->value(row, 4);
        if (options.include_text) {
            chunk.text = result->value(row, 5);
            chunk.context = result->value(row, 6);
        }
        try {
            chunk.metadata = nlohmann::json::parse(result->value(row, 7)).get<Metadata>();
        } catch (...) {
            return fail<std::vector<StoredChunk>>(Errc::corrupt_index,
                                                  "PostgreSQL chunk metadata is invalid");
        }
        chunk.start_line = *start_line;
        chunk.end_line = *end_line;
        if (options.include_embedding && !result->is_null(row, 10)) {
            auto embedding = parse_vector(result->value(row, 10));
            if (!embedding)
                return unexpected(embedding.error());
            chunk.embedding = std::move(*embedding);
        }
        output.push_back(std::move(chunk));
    }
    return output;
}

Result<BackendStats> PostgresBackend::stats() const {
    auto lease = impl_->pool->acquire(impl_->config.acquire_timeout);
    if (!lease)
        return unexpected(lease.error());
    const std::array<std::string, 1> parameters{impl_->config.corpus};
    auto result = lease->connection().execute_prepared("lrs_stats", parameters);
    if (!result)
        return unexpected(result.error());
    BackendStats output;
    output.capabilities = {true, true, true, true, true};
    output.dense_implementation = "pgvector";
    output.dense_algorithm =
        impl_->config.vector_index == PostgresVectorIndex::exact ? "exact" : "hnsw";
    output.dense_exact = impl_->config.vector_index == PostgresVectorIndex::exact;
    if (result->rows() == 0)
        return output;
    auto documents = integer<std::size_t>(result->value(0, 0), "document count");
    auto chunks = integer<std::size_t>(result->value(0, 1), "chunk count");
    auto dimension = integer<std::size_t>(result->value(0, 2), "dimension");
    auto generation = integer<std::uint64_t>(result->value(0, 3), "generation");
    if (!documents || !chunks || !dimension || !generation)
        return fail<BackendStats>(Errc::corrupt_index, "PostgreSQL statistics are invalid");
    output.live_documents = *documents;
    output.live_chunks = *chunks;
    output.embedding_dimension = *dimension;
    output.generation = *generation;
    return output;
}

} // namespace rag::backend
