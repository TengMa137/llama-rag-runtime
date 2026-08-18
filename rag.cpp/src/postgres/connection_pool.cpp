#include "connection_pool.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <limits>

namespace rag::postgres {
namespace {

Errc sql_error_code(const PGresult* result) {
    const char* state = result ? PQresultErrorField(result, PG_DIAG_SQLSTATE) : nullptr;
    if (!state)
        return Errc::unavailable;
    const std::string_view code(state);
    if (code == "23505")
        return Errc::already_exists;
    if (code.starts_with("22"))
        return Errc::invalid_argument;
    return Errc::unavailable;
}

std::string bounded_error(PGconn* connection, PGresult* result) {
    constexpr std::size_t limit = 1024;
    const char* raw = result ? PQresultErrorMessage(result) : PQerrorMessage(connection);
    std::string message = raw ? raw : "PostgreSQL operation failed";
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r'))
        message.pop_back();
    if (message.size() > limit)
        message.resize(limit);
    return message.empty() ? "PostgreSQL operation failed" : message;
}

} // namespace

QueryResult::~QueryResult() {
    if (result_)
        PQclear(result_);
}

QueryResult::QueryResult(QueryResult&& other) noexcept : result_(other.result_) {
    other.result_ = nullptr;
}

QueryResult& QueryResult::operator=(QueryResult&& other) noexcept {
    if (this == &other)
        return *this;
    if (result_)
        PQclear(result_);
    result_ = other.result_;
    other.result_ = nullptr;
    return *this;
}

std::size_t QueryResult::rows() const noexcept {
    return result_ ? static_cast<std::size_t>(PQntuples(result_)) : 0;
}

std::size_t QueryResult::columns() const noexcept {
    return result_ ? static_cast<std::size_t>(PQnfields(result_)) : 0;
}

bool QueryResult::is_null(std::size_t row, std::size_t column) const noexcept {
    return !result_ || row >= rows() || column >= columns() ||
           PQgetisnull(result_, static_cast<int>(row), static_cast<int>(column)) != 0;
}

std::string_view QueryResult::value(std::size_t row, std::size_t column) const noexcept {
    if (is_null(row, column))
        return {};
    return {PQgetvalue(result_, static_cast<int>(row), static_cast<int>(column)),
            static_cast<std::size_t>(
                PQgetlength(result_, static_cast<int>(row), static_cast<int>(column)))};
}

std::uint64_t QueryResult::affected_rows() const noexcept {
    if (!result_)
        return 0;
    const char* value = PQcmdTuples(result_);
    if (!value || *value == '\0')
        return 0;
    std::uint64_t output = 0;
    const auto parsed =
        std::from_chars(value, value + std::char_traits<char>::length(value), output);
    return parsed.ec == std::errc{} ? output : 0;
}

Connection::~Connection() {
    if (connection_)
        PQfinish(connection_);
}

Connection::Connection(Connection&& other) noexcept
    : connection_(other.connection_), prepared_(std::move(other.prepared_)) {
    other.connection_ = nullptr;
}

Connection& Connection::operator=(Connection&& other) noexcept {
    if (this == &other)
        return *this;
    if (connection_)
        PQfinish(connection_);
    connection_ = other.connection_;
    prepared_ = std::move(other.prepared_);
    other.connection_ = nullptr;
    return *this;
}

Result<Connection> Connection::open(const backend::PostgresConfig& config) {
    if (auto valid = backend::validate_postgres_config(config); !valid)
        return unexpected(valid.error());
    PGconn* connection = PQconnectdb(config.connection_string.c_str());
    if (!connection || PQstatus(connection) != CONNECTION_OK) {
        const std::string error = bounded_error(connection, nullptr);
        if (connection)
            PQfinish(connection);
        return fail<Connection>(Errc::unavailable, error);
    }
    Connection output(connection);
    const std::string timeout =
        "SET statement_timeout = " + std::to_string(config.statement_timeout.count());
    if (auto set = output.execute(timeout); !set)
        return unexpected(set.error());
    if (auto set = output.execute("SET idle_in_transaction_session_timeout = 30000"); !set)
        return unexpected(set.error());
    return output;
}

Result<QueryResult> Connection::checked(PGresult* result) {
    if (!result)
        return fail<QueryResult>(Errc::unavailable, bounded_error(connection_, nullptr));
    const auto status = PQresultStatus(result);
    if (status != PGRES_COMMAND_OK && status != PGRES_TUPLES_OK) {
        const auto code = sql_error_code(result);
        const auto message = bounded_error(connection_, result);
        PQclear(result);
        return fail<QueryResult>(code, message);
    }
    return QueryResult(result);
}

Result<QueryResult> Connection::execute(std::string_view sql) {
    const std::string owned(sql);
    return checked(PQexec(connection_, owned.c_str()));
}

Result<QueryResult> Connection::execute_params(std::string_view sql,
                                               std::span<const std::string> parameters) {
    const std::string owned(sql);
    std::vector<const char*> values;
    values.reserve(parameters.size());
    for (const auto& parameter : parameters)
        values.push_back(parameter.c_str());
    return checked(PQexecParams(connection_, owned.c_str(), static_cast<int>(values.size()),
                                nullptr, values.data(), nullptr, nullptr, 0));
}

Result<void> Connection::prepare(std::string_view name, std::string_view sql,
                                 std::size_t parameters) {
    const std::string owned_name(name);
    const std::string owned_sql(sql);
    const auto existing = prepared_.find(owned_name);
    if (existing != prepared_.end()) {
        if (existing->second == owned_sql)
            return {};
        return fail<void>(Errc::invalid_argument, "PostgreSQL prepared statement name was reused");
    }
    auto result = checked(PQprepare(connection_, owned_name.c_str(), owned_sql.c_str(),
                                    static_cast<int>(parameters), nullptr));
    if (!result)
        return unexpected(result.error());
    prepared_.emplace(owned_name, owned_sql);
    return {};
}

Result<QueryResult> Connection::execute_prepared(std::string_view name,
                                                 std::span<const std::string> parameters) {
    const std::string owned_name(name);
    std::vector<const char*> values;
    values.reserve(parameters.size());
    for (const auto& parameter : parameters)
        values.push_back(parameter.c_str());
    return checked(PQexecPrepared(connection_, owned_name.c_str(), static_cast<int>(values.size()),
                                  values.data(), nullptr, nullptr, 0));
}

Result<void> Connection::cancel() noexcept {
    PGcancel* cancel = connection_ ? PQgetCancel(connection_) : nullptr;
    if (!cancel)
        return fail<void>(Errc::unavailable, "PostgreSQL cancellation is unavailable");
    char error[256]{};
    const int cancelled = PQcancel(cancel, error, sizeof(error));
    PQfreeCancel(cancel);
    if (cancelled == 0)
        return fail<void>(Errc::unavailable, "PostgreSQL cancellation failed");
    return {};
}

ConnectionPool::Lease::~Lease() { reset(); }

ConnectionPool::Lease::Lease(Lease&& other) noexcept : pool_(other.pool_), index_(other.index_) {
    other.pool_ = nullptr;
}

ConnectionPool::Lease& ConnectionPool::Lease::operator=(Lease&& other) noexcept {
    if (this == &other)
        return *this;
    reset();
    pool_ = other.pool_;
    index_ = other.index_;
    other.pool_ = nullptr;
    return *this;
}

Connection& ConnectionPool::Lease::connection() const noexcept {
    return const_cast<Connection&>(pool_->connections_[index_]);
}

void ConnectionPool::Lease::reset() noexcept {
    if (pool_)
        pool_->release(index_);
    pool_ = nullptr;
}

Result<std::shared_ptr<ConnectionPool>> ConnectionPool::open(const backend::PostgresConfig& config,
                                                             Initializer initializer,
                                                             void* context) {
    auto output = std::make_shared<ConnectionPool>();
    output->connections_.reserve(config.pool_size);
    for (std::size_t index = 0; index < config.pool_size; ++index) {
        auto connection = Connection::open(config);
        if (!connection)
            return unexpected(connection.error());
        if (initializer) {
            if (auto initialized = initializer(*connection, context); !initialized)
                return unexpected(initialized.error());
        }
        output->connections_.push_back(std::move(*connection));
        output->available_.push_back(index);
    }
    return output;
}

Result<ConnectionPool::Lease> ConnectionPool::acquire(std::chrono::milliseconds timeout) const {
    std::unique_lock lock(mutex_);
    if (!ready_.wait_for(lock, timeout, [&] { return !available_.empty(); }))
        return fail<Lease>(Errc::unavailable, "PostgreSQL connection pool is saturated");
    const auto index = available_.back();
    available_.pop_back();
    return Lease(this, index);
}

void ConnectionPool::release(std::size_t index) const noexcept {
    {
        std::lock_guard lock(mutex_);
        available_.push_back(index);
    }
    ready_.notify_one();
}

void ConnectionPool::cancel_all() noexcept {
    for (auto& connection : connections_)
        (void)connection.cancel();
}

} // namespace rag::postgres
