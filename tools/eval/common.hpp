#pragma once

#include <cstdint>
#include <vector>

namespace lrs::eval {

[[nodiscard]] double percentile(std::vector<double> values, double quantile);
[[nodiscard]] std::uint64_t process_rss_bytes();

} // namespace lrs::eval
