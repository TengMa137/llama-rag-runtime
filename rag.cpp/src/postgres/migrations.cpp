#include "migrations.hpp"

#include <array>
#include <cstdint>

namespace rag::postgres {

std::string qualified_schema(const std::string& schema) { return '"' + schema + '"'; }

std::string migration_sql(const std::string& schema, std::uint32_t version) {
    const std::string q = qualified_schema(schema);
    if (version == 1) {
        return "CREATE EXTENSION IF NOT EXISTS vector;"
               "CREATE SCHEMA IF NOT EXISTS " +
               q +
               ";"
               "CREATE TABLE IF NOT EXISTS " +
               q +
               ".schema_migrations("
               "version integer PRIMARY KEY,"
               "applied_at timestamptz NOT NULL DEFAULT clock_timestamp());";
    }
    if (version == 2) {
        return "CREATE TABLE IF NOT EXISTS " + q +
               ".corpora("
               "id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,"
               "name text NOT NULL UNIQUE,"
               "dimension integer NOT NULL CHECK(dimension > 0),"
               "metric text NOT NULL CHECK(metric = 'cosine'),"
               "embedding_identity text NOT NULL,"
               "chunking_fingerprint text NOT NULL,"
               "generation bigint NOT NULL DEFAULT 0 CHECK(generation >= 0));"
               "CREATE TABLE IF NOT EXISTS " +
               q +
               ".document_revisions("
               "corpus_id bigint NOT NULL REFERENCES " +
               q +
               ".corpora(id) ON DELETE CASCADE,"
               "external_id text NOT NULL,"
               "revision bigint NOT NULL CHECK(revision > 0),"
               "deleted boolean NOT NULL,"
               "content_hash text NOT NULL,"
               "PRIMARY KEY(corpus_id, external_id));"
               "CREATE TABLE IF NOT EXISTS " +
               q +
               ".documents("
               "corpus_id bigint NOT NULL REFERENCES " +
               q +
               ".corpora(id) ON DELETE CASCADE,"
               "external_id text NOT NULL,"
               "revision bigint NOT NULL CHECK(revision > 0),"
               "title text NOT NULL,"
               "content text NOT NULL,"
               "metadata jsonb NOT NULL,"
               "content_hash text NOT NULL,"
               "PRIMARY KEY(corpus_id, external_id));"
               "CREATE TABLE IF NOT EXISTS " +
               q +
               ".chunks("
               "corpus_id bigint NOT NULL,"
               "document_id text NOT NULL,"
               "document_revision bigint NOT NULL CHECK(document_revision > 0),"
               "chunk_key text NOT NULL,"
               "ordinal integer NOT NULL CHECK(ordinal >= 0),"
               "text text NOT NULL,"
               "indexed_text text NOT NULL,"
               "context text NOT NULL,"
               "start_line integer NOT NULL CHECK(start_line >= 0),"
               "end_line integer NOT NULL CHECK(end_line >= start_line),"
               "token_count integer NOT NULL CHECK(token_count >= 0),"
               "embedding vector NOT NULL,"
               "PRIMARY KEY(corpus_id, chunk_key),"
               "UNIQUE(corpus_id, document_id, ordinal),"
               "FOREIGN KEY(corpus_id, document_id) REFERENCES " +
               q +
               ".documents(corpus_id, external_id) ON DELETE CASCADE);"
               "CREATE TABLE IF NOT EXISTS " +
               q +
               ".terms("
               "corpus_id bigint NOT NULL REFERENCES " +
               q +
               ".corpora(id) ON DELETE CASCADE,"
               "term text NOT NULL,"
               "PRIMARY KEY(corpus_id, term));"
               "CREATE TABLE IF NOT EXISTS " +
               q +
               ".postings("
               "corpus_id bigint NOT NULL,"
               "term text NOT NULL,"
               "chunk_key text NOT NULL,"
               "term_frequency integer NOT NULL CHECK(term_frequency > 0),"
               "PRIMARY KEY(corpus_id, term, chunk_key),"
               "FOREIGN KEY(corpus_id, term) REFERENCES " +
               q +
               ".terms(corpus_id, term) ON DELETE CASCADE,"
               "FOREIGN KEY(corpus_id, chunk_key) REFERENCES " +
               q +
               ".chunks(corpus_id, chunk_key) ON DELETE CASCADE);"
               "CREATE TABLE IF NOT EXISTS " +
               q +
               ".index_jobs("
               "job_id text PRIMARY KEY,"
               "corpus_name text NOT NULL,"
               "document_id text NOT NULL,"
               "revision bigint NOT NULL CHECK(revision > 0),"
               "operation text NOT NULL,"
               "status text NOT NULL,"
               "content_hash text NOT NULL,"
               "record jsonb NOT NULL,"
               "mutation_applied boolean,"
               "error_code integer,"
               "error_summary text,"
               "created_at_ms bigint NOT NULL,"
               "updated_at_ms bigint NOT NULL);"
               "CREATE INDEX IF NOT EXISTS documents_metadata_gin ON " +
               q +
               ".documents USING gin(metadata jsonb_path_ops);"
               "CREATE INDEX IF NOT EXISTS postings_term_lookup ON " +
               q +
               ".postings(corpus_id, term);"
               "CREATE INDEX IF NOT EXISTS chunks_document_lookup ON " +
               q +
               ".chunks(corpus_id, document_id, ordinal);"
               "CREATE INDEX IF NOT EXISTS jobs_document_revision ON " +
               q + ".index_jobs(corpus_name, document_id, revision DESC);";
    }
    if (version == 3) {
        return "CREATE TABLE IF NOT EXISTS " + q +
               ".migration_runs("
               "corpus_name text NOT NULL,"
               "run_id text NOT NULL,"
               "source_fingerprint text NOT NULL,"
               "last_document text NOT NULL,"
               "document_count bigint NOT NULL CHECK(document_count >= 0),"
               "chunk_count bigint NOT NULL CHECK(chunk_count >= 0),"
               "complete boolean NOT NULL,"
               "updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),"
               "PRIMARY KEY(corpus_name, run_id));";
    }
    return {};
}

Result<void> migrate(Connection& connection, const std::string& schema) {
    const std::string lock_key = "lrs_rag_schema:" + schema;
    const std::array<std::string, 1> lock_parameters{lock_key};
    auto locked = connection.execute_params("SELECT pg_advisory_lock(hashtextextended($1, 0))",
                                            lock_parameters);
    if (!locked)
        return unexpected(locked.error());

    auto unlock = [&] {
        (void)connection.execute_params("SELECT pg_advisory_unlock(hashtextextended($1, 0))",
                                        lock_parameters);
    };
    if (auto first = connection.execute(migration_sql(schema, 1)); !first) {
        unlock();
        return unexpected(first.error());
    }
    const std::string q = qualified_schema(schema);
    auto applied =
        connection.execute("SELECT version FROM " + q + ".schema_migrations ORDER BY version");
    if (!applied) {
        unlock();
        return unexpected(applied.error());
    }
    std::array<bool, 4> present{};
    for (std::size_t row = 0; row < applied->rows(); ++row) {
        const auto value = applied->value(row, 0);
        if (value == "1")
            present[1] = true;
        else if (value == "2")
            present[2] = true;
        else if (value == "3")
            present[3] = true;
        else {
            unlock();
            return fail<void>(Errc::corrupt_index, "unknown PostgreSQL schema migration");
        }
    }
    for (std::uint32_t version = 1; version <= 3; ++version) {
        if (present[version])
            continue;
        if (auto begun = connection.execute("BEGIN"); !begun) {
            unlock();
            return unexpected(begun.error());
        }
        if (version != 1) {
            if (auto changed = connection.execute(migration_sql(schema, version)); !changed) {
                (void)connection.execute("ROLLBACK");
                unlock();
                return unexpected(changed.error());
            }
        }
        const std::array<std::string, 1> parameter{std::to_string(version)};
        if (auto recorded = connection.execute_params(
                "INSERT INTO " + q + ".schema_migrations(version) VALUES($1::integer)", parameter);
            !recorded) {
            (void)connection.execute("ROLLBACK");
            unlock();
            return unexpected(recorded.error());
        }
        if (auto committed = connection.execute("COMMIT"); !committed) {
            (void)connection.execute("ROLLBACK");
            unlock();
            return unexpected(committed.error());
        }
    }
    unlock();
    return {};
}

} // namespace rag::postgres
