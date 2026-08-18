#include "rag/ingestion/postgres_job_store.hpp"

#include <array>

#include "../postgres/connection_pool.hpp"
#include "../postgres/migrations.hpp"

namespace rag::ingestion {
namespace {

Result<void> prepare_jobs(postgres::Connection& connection, void* opaque) {
    const auto& schema = *static_cast<const std::string*>(opaque);
    const auto q = postgres::qualified_schema(schema);
    const std::string put =
        "INSERT INTO " + q +
        ".index_jobs(job_id,corpus_name,document_id,revision,operation,status,content_hash,record,"
        "mutation_applied,error_code,error_summary,created_at_ms,updated_at_ms) VALUES("
        "$1,$2,$3,$4::bigint,$5,$6,$7,$8::jsonb,"
        "CASE WHEN $8::jsonb ? 'mutation_applied' THEN ($8::jsonb->>'mutation_applied')::boolean "
        "ELSE NULL END,"
        "CASE WHEN $8::jsonb ? 'error' THEN ($8::jsonb#>>'{error,code}')::integer ELSE NULL END,"
        "COALESCE($8::jsonb#>>'{error,message}',''),$9::bigint,$10::bigint) "
        "ON CONFLICT(job_id) DO UPDATE SET corpus_name=EXCLUDED.corpus_name,"
        "document_id=EXCLUDED.document_id,revision=EXCLUDED.revision,operation=EXCLUDED.operation,"
        "status=EXCLUDED.status,content_hash=EXCLUDED.content_hash,record=EXCLUDED.record,"
        "mutation_applied=EXCLUDED.mutation_applied,error_code=EXCLUDED.error_code,"
        "error_summary=EXCLUDED.error_summary,created_at_ms=EXCLUDED.created_at_ms,"
        "updated_at_ms=EXCLUDED.updated_at_ms";
    if (auto prepared = connection.prepare("lrs_job_put", put, 10); !prepared)
        return prepared;
    return connection.prepare("lrs_jobs_load",
                              "SELECT record::text FROM " + q +
                                  ".index_jobs WHERE corpus_name=$1 ORDER BY created_at_ms,job_id",
                              1);
}

} // namespace

struct PostgresJobStore::Impl {
    backend::PostgresConfig config;
    std::shared_ptr<postgres::ConnectionPool> pool;
};

PostgresJobStore::PostgresJobStore(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

PostgresJobStore::~PostgresJobStore() {
    if (impl_ && impl_->pool)
        impl_->pool->cancel_all();
}

Result<std::shared_ptr<PostgresJobStore>> PostgresJobStore::open(backend::PostgresConfig config) {
    if (auto valid = backend::validate_postgres_config(config); !valid)
        return unexpected(valid.error());
    auto bootstrap = postgres::Connection::open(config);
    if (!bootstrap)
        return unexpected(bootstrap.error());
    if (config.run_migrations) {
        if (auto migrated = postgres::migrate(*bootstrap, config.schema); !migrated)
            return unexpected(migrated.error());
    }
    // Job persistence is serialized by the coordinator and needs one bounded
    // connection independent of the candidate-query pool.
    config.pool_size = 1;
    auto pool = postgres::ConnectionPool::open(config, prepare_jobs, &config.schema);
    if (!pool)
        return unexpected(pool.error());
    auto impl = std::make_unique<Impl>();
    impl->config = std::move(config);
    impl->pool = std::move(*pool);
    return std::shared_ptr<PostgresJobStore>(new PostgresJobStore(std::move(impl)));
}

Result<void> PostgresJobStore::persist(const IngestionJob& job) {
    auto record = serialize_ingestion_job(job);
    if (!record)
        return unexpected(record.error());
    auto lease = impl_->pool->acquire(impl_->config.acquire_timeout);
    if (!lease)
        return unexpected(lease.error());
    const std::array<std::string, 10> parameters{job.id,
                                                 impl_->config.corpus,
                                                 job.input.document,
                                                 std::to_string(job.revision),
                                                 job.operation == JobOperation::upsert ? "upsert"
                                                                                       : "erase",
                                                 std::string(name(job.status)),
                                                 job.content_hash,
                                                 std::move(*record),
                                                 std::to_string(job.created_at_ms),
                                                 std::to_string(job.updated_at_ms)};
    auto saved = lease->connection().execute_prepared("lrs_job_put", parameters);
    if (!saved)
        return unexpected(saved.error());
    return {};
}

Result<std::vector<IngestionJob>> PostgresJobStore::load_latest() const {
    auto lease = impl_->pool->acquire(impl_->config.acquire_timeout);
    if (!lease)
        return unexpected(lease.error());
    const std::array<std::string, 1> parameters{impl_->config.corpus};
    auto rows = lease->connection().execute_prepared("lrs_jobs_load", parameters);
    if (!rows)
        return unexpected(rows.error());
    std::vector<IngestionJob> output;
    output.reserve(rows->rows());
    for (std::size_t row = 0; row < rows->rows(); ++row) {
        auto job = deserialize_ingestion_job(rows->value(row, 0));
        if (!job)
            return unexpected(job.error());
        output.push_back(std::move(*job));
    }
    return output;
}

} // namespace rag::ingestion
