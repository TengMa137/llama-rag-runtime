#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"

namespace rag::store {

// Validated offsets into one immutable .ragdb byte sequence. The layout owns
// no bytes; callers must keep the source sequence alive while using ranges.
struct SectionRange {
    std::uint32_t tag = 0;
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
};

struct ContainerLayout {
    std::uint16_t format_major = 0;
    std::uint16_t format_minor = 0;
    std::uint32_t flags = 0;
    std::vector<SectionRange> sections;
};

// Verifies magic, version, CRC, unique tags, bounds, and non-overlap without
// copying section payloads.
[[nodiscard]] Result<ContainerLayout> validate_container_layout(std::string_view bytes);

} // namespace rag::store
