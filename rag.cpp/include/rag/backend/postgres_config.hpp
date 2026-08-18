#pragma once

#include <chrono>
#include <cstddef>
#include <string>

#include "rag/core/types.hpp"

namespace rag::backend {

enum class PostgresVectorIndex { exact, hnsw };

struct PostgresConfig {
    std::string connection_string;
    std::string schema = "lrs_rag";
    std::string corpus = "default";
    std::size_t pool_size = 4;
    std::chrono::milliseconds acquire_timeout{5'000};
    std::chrono::milliseconds statement_timeout{10'000};
    PostgresVectorIndex vector_index = PostgresVectorIndex::exact;
    std::size_t hnsw_ef_search = 64;
    bool run_migrations = true;
};

[[nodiscard]] Result<void> validate_postgres_config(const PostgresConfig& config);

} // namespace rag::backend
