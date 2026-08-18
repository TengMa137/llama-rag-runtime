#include "rag/backend/postgres_config.hpp"

#include <algorithm>
#include <cctype>
#include <string_view>

#include <libpq-fe.h>

namespace rag::backend {
namespace {

bool identifier(std::string_view value) {
    if (value.empty() || value.size() > 63 ||
        !(std::isalpha(static_cast<unsigned char>(value.front())) || value.front() == '_'))
        return false;
    return std::all_of(value.begin() + 1, value.end(),
                       [](unsigned char byte) { return std::isalnum(byte) || byte == '_'; });
}

bool loopback_host(std::string_view host) {
    return host.empty() || host == "localhost" || host == "::1" || host.front() == '/' ||
           host.starts_with("127.");
}

} // namespace

Result<void> validate_postgres_config(const PostgresConfig& config) {
    if (config.connection_string.empty())
        return fail<void>(Errc::invalid_argument, "PostgreSQL connection string is required");
    if (!identifier(config.schema) || config.corpus.empty() || config.corpus.size() > 256)
        return fail<void>(Errc::invalid_argument, "PostgreSQL schema or corpus is invalid");
    if (config.pool_size == 0 || config.pool_size > 32 || config.acquire_timeout.count() <= 0 ||
        config.statement_timeout.count() <= 0 || config.statement_timeout.count() > 600'000 ||
        config.hnsw_ef_search == 0 || config.hnsw_ef_search > 10'000)
        return fail<void>(Errc::invalid_argument, "PostgreSQL pool or timeout policy is invalid");

    char* parse_error = nullptr;
    PQconninfoOption* options = PQconninfoParse(config.connection_string.c_str(), &parse_error);
    if (!options) {
        if (parse_error)
            PQfreemem(parse_error);
        return fail<void>(Errc::invalid_argument, "PostgreSQL connection string is invalid");
    }
    std::string host;
    std::string hostaddr;
    std::string sslmode;
    for (auto* option = options; option->keyword != nullptr; ++option) {
        const std::string_view keyword(option->keyword);
        const std::string_view value = option->val ? option->val : "";
        if (keyword == "host")
            host = value;
        else if (keyword == "hostaddr")
            hostaddr = value;
        else if (keyword == "sslmode")
            sslmode = value;
    }
    PQconninfoFree(options);
    const auto all_loopback = [](std::string_view hosts) {
        std::size_t start = 0;
        while (start <= hosts.size()) {
            const auto end = hosts.find(',', start);
            if (!loopback_host(hosts.substr(
                    start, end == std::string_view::npos ? hosts.size() - start : end - start)))
                return false;
            if (end == std::string_view::npos)
                break;
            start = end + 1;
        }
        return true;
    };
    const bool local = all_loopback(host) && all_loopback(hostaddr);
    if (!local && sslmode != "verify-ca" && sslmode != "verify-full")
        return fail<void>(Errc::invalid_argument,
                          "non-loopback PostgreSQL connections require certificate-verifying TLS");
    return {};
}

} // namespace rag::backend
