#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <libpq-fe.h>

#include "rag/backend/postgres_config.hpp"

namespace rag::postgres {

class QueryResult {
  public:
    QueryResult() = default;
    explicit QueryResult(PGresult* result) : result_(result) {}
    ~QueryResult();
    QueryResult(const QueryResult&) = delete;
    QueryResult& operator=(const QueryResult&) = delete;
    QueryResult(QueryResult&& other) noexcept;
    QueryResult& operator=(QueryResult&& other) noexcept;

    [[nodiscard]] std::size_t rows() const noexcept;
    [[nodiscard]] std::size_t columns() const noexcept;
    [[nodiscard]] bool is_null(std::size_t row, std::size_t column) const noexcept;
    [[nodiscard]] std::string_view value(std::size_t row, std::size_t column) const noexcept;
    [[nodiscard]] std::uint64_t affected_rows() const noexcept;

  private:
    PGresult* result_ = nullptr;
};

class Connection {
  public:
    Connection() = default;
    ~Connection();
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    Connection(Connection&& other) noexcept;
    Connection& operator=(Connection&& other) noexcept;

    [[nodiscard]] static Result<Connection> open(const backend::PostgresConfig& config);
    [[nodiscard]] Result<QueryResult> execute(std::string_view sql);
    [[nodiscard]] Result<QueryResult> execute_params(std::string_view sql,
                                                     std::span<const std::string> parameters);
    [[nodiscard]] Result<void> prepare(std::string_view name, std::string_view sql,
                                       std::size_t parameters);
    [[nodiscard]] Result<QueryResult> execute_prepared(std::string_view name,
                                                       std::span<const std::string> parameters);
    [[nodiscard]] Result<void> cancel() noexcept;
    [[nodiscard]] PGconn* native() const noexcept { return connection_; }

  private:
    explicit Connection(PGconn* connection) : connection_(connection) {}
    [[nodiscard]] Result<QueryResult> checked(PGresult* result);
    PGconn* connection_ = nullptr;
    std::unordered_map<std::string, std::string> prepared_;
};

class ConnectionPool {
  public:
    using Initializer = Result<void> (*)(Connection&, void*);

    class Lease {
      public:
        Lease() = default;
        ~Lease();
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        [[nodiscard]] Connection& connection() const noexcept;
        [[nodiscard]] Connection* operator->() const noexcept { return &connection(); }

      private:
        friend class ConnectionPool;
        Lease(const ConnectionPool* pool, std::size_t index) : pool_(pool), index_(index) {}
        void reset() noexcept;
        const ConnectionPool* pool_ = nullptr;
        std::size_t index_ = 0;
    };

    [[nodiscard]] static Result<std::shared_ptr<ConnectionPool>>
    open(const backend::PostgresConfig& config, Initializer initializer, void* context);
    [[nodiscard]] Result<Lease> acquire(std::chrono::milliseconds timeout) const;
    void cancel_all() noexcept;

  private:
    void release(std::size_t index) const noexcept;
    std::vector<Connection> connections_;
    mutable std::mutex mutex_;
    mutable std::condition_variable ready_;
    mutable std::vector<std::size_t> available_;
};

} // namespace rag::postgres
