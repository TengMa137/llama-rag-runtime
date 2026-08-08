#pragma once
// Device-selection seam retained for Corpus routing. The owned local-agent
// build is CPU-only, so every operation declines without touching its output.

#include <cstddef>
#include <span>
#include <string>

namespace rag::gpu {
enum class Backend { none };
[[nodiscard]] const char* backend_name(Backend backend) noexcept;
struct DeviceInfo {
    Backend backend = Backend::none;
    std::string name = "cpu";
    bool unified_memory = false;
    std::size_t max_buffer_bytes = 0;
};
[[nodiscard]] bool available() noexcept;
[[nodiscard]] const DeviceInfo& device_info() noexcept;
void disable() noexcept;
[[nodiscard]] bool score_batch(std::span<const float> corpus, std::span<const float> queries,
                               std::size_t dimension, std::span<float> output) noexcept;
[[nodiscard]] std::size_t min_batch_work() noexcept;
} // namespace rag::gpu
