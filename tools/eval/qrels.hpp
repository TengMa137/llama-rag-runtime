#pragma once

#include <string>

#include <nlohmann/json.hpp>

#include <rag/core/types.hpp>

namespace lrs::eval {

[[nodiscard]] rag::Result<nlohmann::json> run_qrels(const std::string& path);

} // namespace lrs::eval
