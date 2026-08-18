#pragma once

#include <string>

#include "connection_pool.hpp"

namespace rag::postgres {

[[nodiscard]] Result<void> migrate(Connection& connection, const std::string& schema);
[[nodiscard]] std::string qualified_schema(const std::string& schema);
[[nodiscard]] std::string migration_sql(const std::string& schema, std::uint32_t version);

} // namespace rag::postgres
