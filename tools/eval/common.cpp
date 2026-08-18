#include "common.hpp"

#include <algorithm>
#include <fstream>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace lrs::eval {

double percentile(std::vector<double> values, double quantile) {
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const double position = quantile * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] + fraction * (values[upper] - values[lower]);
}

std::uint64_t process_rss_bytes() {
#if defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info),
                  &count) == KERN_SUCCESS)
        return info.resident_size;
#elif defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    std::uint64_t total_pages = 0;
    std::uint64_t resident_pages = 0;
    if (statm >> total_pages >> resident_pages)
        return resident_pages * static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
#endif
    return 0;
}

} // namespace lrs::eval
