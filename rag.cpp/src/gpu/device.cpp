#include "rag/gpu/device.hpp"

namespace rag::gpu {
const char* backend_name(Backend) noexcept { return "none"; }
bool available() noexcept { return false; }
const DeviceInfo& device_info() noexcept {
    static const DeviceInfo cpu;
    return cpu;
}
void disable() noexcept {}
bool score_batch(std::span<const float>, std::span<const float>, std::size_t,
                 std::span<float>) noexcept {
    return false;
}
std::size_t min_batch_work() noexcept { return static_cast<std::size_t>(-1); }
} // namespace rag::gpu
