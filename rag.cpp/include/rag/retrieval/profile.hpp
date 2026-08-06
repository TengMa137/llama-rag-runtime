#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace rag::retrieval {

enum class Profile { efficiency, balanced, quality };

struct ProfileOverrides {
    std::optional<std::size_t> candidate_pool;
    std::optional<std::size_t> hnsw_threshold;
    std::optional<float> mmr_lambda;
    std::optional<std::size_t> adjacent_line_gap;
    std::optional<bool> diversity;
    std::optional<bool> context_stitching;
};

struct SearchOptions {
    std::size_t top_k = 10;
    Profile profile = Profile::balanced;
    ProfileOverrides overrides{};
};

struct Diagnostics {
    Profile profile = Profile::balanced;
    std::vector<std::string> stages;
    std::vector<std::string> fallback_reasons;
    std::size_t candidate_pool = 0;
    std::size_t result_count = 0;
    double elapsed_ms = 0.0;
};

[[nodiscard]] inline const char* name(Profile profile) noexcept {
    switch (profile) {
        case Profile::efficiency:
            return "efficiency";
        case Profile::balanced:
            return "balanced";
        case Profile::quality:
            return "quality";
    }
    return "balanced";
}

} // namespace rag::retrieval
