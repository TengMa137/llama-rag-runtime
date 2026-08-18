#pragma once

#include <memory>

#include "rag/backend/postgres_config.hpp"
#include "rag/migration/contract.hpp"

namespace rag::migration {

[[nodiscard]] Result<std::unique_ptr<Endpoint>>
open_postgres_source(backend::PostgresConfig config);
[[nodiscard]] Result<std::unique_ptr<Endpoint>>
open_postgres_destination(backend::PostgresConfig config);

} // namespace rag::migration
