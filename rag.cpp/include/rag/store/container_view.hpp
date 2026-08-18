#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/store/container_layout.hpp"
#include "rag/store/format.hpp"

namespace rag::store {

// Read-only, move-only view of a validated memory-mapped .ragdb file. Returned
// section views remain valid only for this object's lifetime.
class ContainerView {
  public:
    ContainerView() = default;
    ~ContainerView();
    ContainerView(const ContainerView&) = delete;
    ContainerView& operator=(const ContainerView&) = delete;
    ContainerView(ContainerView&& other) noexcept;
    ContainerView& operator=(ContainerView&& other) noexcept;

    [[nodiscard]] static Result<ContainerView> open_file(const std::string& path);

    [[nodiscard]] bool has(Tag tag) const;
    [[nodiscard]] std::optional<std::string_view> get(Tag tag) const;
    [[nodiscard]] std::optional<std::string_view> get_raw(std::uint32_t tag) const;
    [[nodiscard]] std::span<const SectionRange> sections() const noexcept { return sections_; }
    [[nodiscard]] std::uint32_t flags() const noexcept { return flags_; }
    [[nodiscard]] std::uint16_t format_major() const noexcept { return major_; }
    [[nodiscard]] std::uint16_t format_minor() const noexcept { return minor_; }
    [[nodiscard]] std::size_t mapped_bytes() const noexcept { return size_; }

  private:
    void reset() noexcept;

    const char* data_ = nullptr;
    std::size_t size_ = 0;
    std::vector<SectionRange> sections_;
    std::uint32_t flags_ = 0;
    std::uint16_t major_ = 0;
    std::uint16_t minor_ = 0;
};

} // namespace rag::store
