#include "rag/migration/postgres_endpoint.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "../postgres/connection_pool.hpp"
#include "../postgres/migrations.hpp"
#include "rag/backend/postgres_backend.hpp"

namespace rag::migration {
namespace {

template <class Integer> Result<Integer> integer(std::string_view value, std::string_view field) {
    Integer output{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), output);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        return fail<Integer>(Errc::corrupt_index,
                             "PostgreSQL migration returned an invalid " + std::string(field));
    return output;
}

Result<Vector> vector(std::string_view value) {
    if (value.size() < 2 || value.front() != '[' || value.back() != ']')
        return fail<Vector>(Errc::corrupt_index, "PostgreSQL migration vector is invalid");
    value.remove_prefix(1);
    value.remove_suffix(1);
    Vector output;
    while (!value.empty()) {
        const auto separator = value.find(',');
        const auto item = value.substr(0, separator);
        try {
            std::size_t parsed = 0;
            const float component = std::stof(std::string(item), &parsed);
            if (parsed != item.size() || !std::isfinite(component))
                return fail<Vector>(Errc::corrupt_index,
                                    "PostgreSQL migration vector component is invalid");
            output.push_back(component);
        } catch (...) {
            return fail<Vector>(Errc::corrupt_index,
                                "PostgreSQL migration vector component is invalid");
        }
        if (separator == std::string_view::npos)
            break;
        value.remove_prefix(separator + 1);
    }
    return output;
}

std::string vector_text(VectorView value) {
    std::ostringstream output;
    output << '[' << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t index = 0; index < value.size(); ++index) {
        if (index != 0)
            output << ',';
        output << value[index];
    }
    output << ']';
    return output.str();
}

struct Corpus {
    std::uint64_t id = 0;
    std::size_t dimension = 0;
    std::string embedding_identity;
    std::string chunking_fingerprint;
};

Result<std::optional<Corpus>> corpus(postgres::Connection& connection,
                                     const backend::PostgresConfig& config) {
    const auto q = postgres::qualified_schema(config.schema);
    const std::array<std::string, 1> parameters{config.corpus};
    auto row = connection.execute_params(
        "SELECT id,dimension,embedding_identity,chunking_fingerprint FROM " + q +
            ".corpora WHERE name=$1",
        parameters);
    if (!row)
        return unexpected(row.error());
    if (row->rows() == 0)
        return std::optional<Corpus>{};
    auto id = integer<std::uint64_t>(row->value(0, 0), "corpus ID");
    auto dimension = integer<std::size_t>(row->value(0, 1), "corpus dimension");
    if (!id || !dimension)
        return fail<std::optional<Corpus>>(Errc::corrupt_index,
                                           "PostgreSQL migration corpus is invalid");
    return std::optional<Corpus>{
        Corpus{*id, *dimension, std::string(row->value(0, 2)), std::string(row->value(0, 3))}};
}

class PostgresEndpoint final : public Endpoint {
  public:
    PostgresEndpoint(backend::PostgresConfig config, postgres::Connection connection,
                     std::shared_ptr<backend::PostgresBackend> backend, bool writable,
                     std::optional<Corpus> corpus)
        : config_(std::move(config)), connection_(std::move(connection)),
          backend_(std::move(backend)), writable_(writable), corpus_(std::move(corpus)) {}

    ~PostgresEndpoint() override {
        if (!writable_)
            (void)connection_.execute("ROLLBACK");
    }

    std::string description() const override {
        return "postgres:" + config_.schema + "/" + config_.corpus;
    }

    Result<DocumentBatch> read_batch(std::string_view after, std::size_t limit) const override {
        if (limit == 0)
            return fail<DocumentBatch>(Errc::invalid_argument, "migration batch limit is zero");
        auto state = current_corpus();
        if (!state)
            return unexpected(state.error());
        if (!*state)
            return DocumentBatch{{}, std::string(after), true};
        const auto q = postgres::qualified_schema(config_.schema);
        const std::array<std::string, 3> parameters{std::to_string((*state)->id),
                                                    std::string(after), std::to_string(limit + 1)};
        auto documents = connection_.execute_params(
            "SELECT external_id,revision,title,content,metadata::text,content_hash FROM " + q +
                ".documents WHERE corpus_id=$1::bigint AND external_id>$2 "
                "ORDER BY external_id LIMIT $3::bigint",
            parameters);
        if (!documents)
            return unexpected(documents.error());
        DocumentBatch output;
        output.complete = documents->rows() <= limit;
        const std::size_t count = std::min(documents->rows(), limit);
        output.documents.reserve(count);
        for (std::size_t row = 0; row < count; ++row) {
            RevisionedDocument record;
            auto revision =
                integer<backend::DocumentRevision>(documents->value(row, 1), "document revision");
            if (!revision)
                return unexpected(revision.error());
            record.revision = *revision;
            auto& document = record.document;
            document.key = documents->value(row, 0);
            document.title = documents->value(row, 2);
            document.content = documents->value(row, 3);
            document.content_hash = documents->value(row, 5);
            document.embedding_identity = (*state)->embedding_identity;
            document.chunking_fingerprint = (*state)->chunking_fingerprint;
            try {
                document.metadata = nlohmann::json::parse(documents->value(row, 4)).get<Metadata>();
            } catch (...) {
                return fail<DocumentBatch>(Errc::corrupt_index,
                                           "PostgreSQL migration metadata is invalid");
            }
            const std::array<std::string, 2> chunk_parameters{std::to_string((*state)->id),
                                                              document.key};
            auto chunks = connection_.execute_params(
                "SELECT chunk_key,ordinal,text,indexed_text,context,start_line,end_line,"
                "embedding::text FROM " +
                    q + ".chunks WHERE corpus_id=$1::bigint AND document_id=$2 ORDER BY ordinal",
                chunk_parameters);
            if (!chunks)
                return unexpected(chunks.error());
            document.chunks.reserve(chunks->rows());
            for (std::size_t chunk_row = 0; chunk_row < chunks->rows(); ++chunk_row) {
                backend::PreparedChunk chunk;
                auto ordinal = integer<std::size_t>(chunks->value(chunk_row, 1), "chunk ordinal");
                auto start = integer<std::uint32_t>(chunks->value(chunk_row, 5), "start line");
                auto end = integer<std::uint32_t>(chunks->value(chunk_row, 6), "end line");
                auto embedding = vector(chunks->value(chunk_row, 7));
                if (!ordinal || !start || !end || !embedding)
                    return fail<DocumentBatch>(Errc::corrupt_index,
                                               "PostgreSQL migration chunk is invalid");
                chunk.key = chunks->value(chunk_row, 0);
                chunk.ordinal = *ordinal;
                chunk.text = chunks->value(chunk_row, 2);
                chunk.indexed_text = chunks->value(chunk_row, 3);
                chunk.context = chunks->value(chunk_row, 4);
                chunk.start_line = *start;
                chunk.end_line = *end;
                chunk.embedding = std::move(*embedding);
                document.chunks.push_back(std::move(chunk));
            }
            output.documents.push_back(std::move(record));
        }
        output.next_cursor =
            output.documents.empty() ? std::string(after) : output.documents.back().document.key;
        return output;
    }

    Result<std::optional<MigrationProgress>> progress(std::string_view run_id) const override {
        if (!writable_)
            return std::optional<MigrationProgress>{};
        const auto q = postgres::qualified_schema(config_.schema);
        const std::array<std::string, 2> parameters{config_.corpus, std::string(run_id)};
        auto result = connection_.execute_params(
            "SELECT source_fingerprint,last_document,document_count,chunk_count,complete FROM " +
                q + ".migration_runs WHERE corpus_name=$1 AND run_id=$2",
            parameters);
        if (!result)
            return unexpected(result.error());
        if (result->rows() == 0)
            return std::optional<MigrationProgress>{};
        auto documents = integer<std::size_t>(result->value(0, 2), "migration document count");
        auto chunks = integer<std::size_t>(result->value(0, 3), "migration chunk count");
        if (!documents || !chunks)
            return fail<std::optional<MigrationProgress>>(
                Errc::corrupt_index, "PostgreSQL migration progress is invalid");
        return std::optional<MigrationProgress>{MigrationProgress{
            std::string(run_id), std::string(result->value(0, 0)), std::string(result->value(0, 1)),
            *documents, *chunks, result->value(0, 4) == "t"}};
    }

    Result<void> write_batch(std::string_view run_id, const CorpusAudit& source,
                             const DocumentBatch& batch,
                             const MigrationProgress& progress_value) override {
        if (!writable_ || !backend_)
            return fail<void>(Errc::invalid_argument, "PostgreSQL migration source is read-only");
        for (const auto& record : batch.documents)
            if (auto activated = backend_->activate(record.document, record.revision); !activated)
                return unexpected(activated.error());
        corpus_.reset();
        return save_progress(run_id, source, progress_value);
    }

    Result<void> finish(std::string_view run_id, const CorpusAudit& source,
                        const MigrationProgress& progress_value) override {
        if (!progress_value.complete)
            return fail<void>(Errc::invalid_argument, "PostgreSQL migration cannot finish early");
        return save_progress(run_id, source, progress_value);
    }

    Result<backend::CandidateList> exact_candidates(VectorView query,
                                                    std::size_t k) const override {
        auto state = current_corpus();
        if (!state)
            return unexpected(state.error());
        if (!*state)
            return backend::CandidateList{};
        const std::string encoded = vector_text(query);
        const auto q = postgres::qualified_schema(config_.schema);
        const std::array<std::string, 3> parameters{std::to_string((*state)->id), encoded,
                                                    std::to_string(k)};
        auto result = connection_.execute_params(
            "SELECT chunk_key,-(embedding <#> $2::vector)::real FROM " + q +
                ".chunks WHERE corpus_id=$1::bigint ORDER BY embedding <#> $2::vector,"
                "chunk_key LIMIT $3::bigint",
            parameters);
        if (!result)
            return unexpected(result.error());
        backend::CandidateList output;
        output.reserve(result->rows());
        for (std::size_t row = 0; row < result->rows(); ++row) {
            float score = 0.0F;
            try {
                score = std::stof(std::string(result->value(row, 1)));
            } catch (...) {
                return fail<backend::CandidateList>(Errc::corrupt_index,
                                                    "PostgreSQL migration score is invalid");
            }
            output.push_back(
                {std::string(result->value(row, 0)), score, backend::ScoreType::cosine});
        }
        return output;
    }

  private:
    Result<std::optional<Corpus>> current_corpus() const {
        if (corpus_)
            return corpus_;
        auto loaded = corpus(connection_, config_);
        if (!loaded)
            return unexpected(loaded.error());
        corpus_ = *loaded;
        return corpus_;
    }

    Result<void> save_progress(std::string_view run_id, const CorpusAudit& source,
                               const MigrationProgress& progress_value) {
        if (progress_value.run_id != run_id ||
            progress_value.source_fingerprint != source.fingerprint)
            return fail<void>(Errc::invalid_argument,
                              "PostgreSQL migration progress is inconsistent");
        const auto q = postgres::qualified_schema(config_.schema);
        const std::array<std::string, 7> parameters{config_.corpus,
                                                    std::string(run_id),
                                                    source.fingerprint,
                                                    progress_value.last_document,
                                                    std::to_string(progress_value.documents),
                                                    std::to_string(progress_value.chunks),
                                                    progress_value.complete ? "true" : "false"};
        auto result = connection_.execute_params(
            "INSERT INTO " + q +
                ".migration_runs(corpus_name,run_id,source_fingerprint,last_document,"
                "document_count,chunk_count,complete) VALUES($1,$2,$3,$4,$5::bigint,$6::bigint,"
                "$7::boolean) ON CONFLICT(corpus_name,run_id) DO UPDATE SET "
                "source_fingerprint=EXCLUDED.source_fingerprint,last_document=EXCLUDED.last_"
                "document,"
                "document_count=EXCLUDED.document_count,chunk_count=EXCLUDED.chunk_count,"
                "complete=EXCLUDED.complete,updated_at=clock_timestamp() WHERE " +
                q + ".migration_runs.source_fingerprint=EXCLUDED.source_fingerprint",
            parameters);
        if (!result)
            return unexpected(result.error());
        if (result->affected_rows() != 1)
            return fail<void>(Errc::already_exists,
                              "PostgreSQL migration progress belongs to another source");
        return {};
    }

    backend::PostgresConfig config_;
    mutable postgres::Connection connection_;
    std::shared_ptr<backend::PostgresBackend> backend_;
    bool writable_ = false;
    mutable std::optional<Corpus> corpus_;
};

Result<std::unique_ptr<Endpoint>> open_postgres(backend::PostgresConfig config, bool writable) {
    config.vector_index = backend::PostgresVectorIndex::exact;
    auto connection = postgres::Connection::open(config);
    if (!connection)
        return unexpected(connection.error());
    if (writable && config.run_migrations)
        if (auto migrated = postgres::migrate(*connection, config.schema); !migrated)
            return unexpected(migrated.error());
    auto state = corpus(*connection, config);
    if (!state)
        return unexpected(state.error());
    if (!writable && !*state)
        return fail<std::unique_ptr<Endpoint>>(Errc::not_found,
                                               "PostgreSQL migration source corpus was not found");
    std::shared_ptr<backend::PostgresBackend> backend;
    if (writable) {
        auto opened = backend::PostgresBackend::open(config);
        if (!opened)
            return unexpected(opened.error());
        backend = std::move(*opened);
    } else if (auto begun = connection->execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY");
               !begun) {
        return unexpected(begun.error());
    }
    return std::unique_ptr<Endpoint>(new PostgresEndpoint(std::move(config), std::move(*connection),
                                                          std::move(backend), writable,
                                                          std::move(*state)));
}

} // namespace

Result<std::unique_ptr<Endpoint>> open_postgres_source(backend::PostgresConfig config) {
    return open_postgres(std::move(config), false);
}

Result<std::unique_ptr<Endpoint>> open_postgres_destination(backend::PostgresConfig config) {
    return open_postgres(std::move(config), true);
}

} // namespace rag::migration
