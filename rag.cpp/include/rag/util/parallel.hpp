#pragma once

// Bounded, operation-local parallel primitives. No process-global pool is
// created: library instances therefore do not leave hidden worker threads
// resident in desktop or mobile hosts.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace rag::util {

inline constexpr std::size_t kParallelMin = 512;

[[nodiscard]] inline std::size_t max_workers() noexcept {
    const unsigned count = std::thread::hardware_concurrency();
    return count ? count : 1;
}

[[nodiscard]] inline std::size_t block_count(std::size_t n,
                                             std::size_t workers = max_workers()) noexcept {
    if (n == 0)
        return 0;
    if (n < kParallelMin || workers <= 1)
        return 1;
    return std::min(n, workers);
}

template <class F>
void parallel_blocks(std::size_t n, F&& body, std::size_t workers = max_workers()) {
    if (n == 0)
        return;
    const std::size_t blocks = block_count(n, workers);
    if (blocks <= 1) {
        body(std::size_t{0}, n, std::size_t{0});
        return;
    }
    const std::size_t chunk = (n + blocks - 1) / blocks;
    std::vector<std::thread> helpers;
    helpers.reserve(blocks - 1);
    for (std::size_t b = 0; b + 1 < blocks; ++b) {
        const std::size_t lo = b * chunk;
        const std::size_t hi = std::min(lo + chunk, n);
        helpers.emplace_back([&, lo, hi, b] { body(lo, hi, b); });
    }
    body((blocks - 1) * chunk, n, blocks - 1);
    for (auto& helper : helpers)
        helper.join();
}

template <class F> void parallel_for(std::size_t n, F&& body, std::size_t workers = max_workers()) {
    parallel_blocks(
        n,
        [&](std::size_t lo, std::size_t hi, std::size_t) {
            for (std::size_t i = lo; i < hi; ++i)
                body(i);
        },
        workers);
}

template <class F> void parallel_for_dynamic(std::size_t n, std::size_t workers, F&& body) {
    if (n == 0)
        return;
    const std::size_t count = std::min(n, std::max<std::size_t>(1, workers));
    if (count <= 1) {
        for (std::size_t i = 0; i < n; ++i)
            body(i);
        return;
    }
    std::atomic<std::size_t> next{0};
    auto drain = [&] {
        for (;;) {
            const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= n)
                return;
            body(i);
        }
    };
    std::vector<std::thread> helpers;
    helpers.reserve(count - 1);
    for (std::size_t i = 1; i < count; ++i)
        helpers.emplace_back(drain);
    drain();
    for (auto& helper : helpers)
        helper.join();
}

} // namespace rag::util
