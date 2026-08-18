#include "rag/store/container_layout.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <new>
#include <set>

#include "rag/store/format.hpp"

namespace rag::store {

Result<ContainerLayout> validate_container_layout(std::string_view bytes) {
    try {
        constexpr std::size_t header_size = 28;
        constexpr std::size_t entry_size = 20;
        constexpr std::size_t trailer_size = 4;
        if (bytes.size() < header_size + trailer_size)
            return fail<ContainerLayout>(Errc::corrupt_index, "too short");

        const auto body = bytes.substr(0, bytes.size() - trailer_size);
        std::uint32_t stored_crc = 0;
        std::memcpy(&stored_crc, bytes.data() + body.size(), trailer_size);
        if (crc32(body) != stored_crc)
            return fail<ContainerLayout>(Errc::corrupt_index, "crc mismatch");

        Reader reader(bytes);
        std::string_view magic;
        if (!reader.bytes(8, magic) || std::memcmp(magic.data(), kMagic, 8) != 0)
            return fail<ContainerLayout>(Errc::corrupt_index, "bad magic");

        ContainerLayout layout;
        if (!reader.u(layout.format_major) || !reader.u(layout.format_minor))
            return fail<ContainerLayout>(Errc::corrupt_index, "version");
        if (layout.format_major != kFormatMajor)
            return fail<ContainerLayout>(
                Errc::corrupt_index, "unsupported format major " +
                                         std::to_string(layout.format_major) +
                                         " (reader supports " + std::to_string(kFormatMajor) + ")");
        std::uint32_t count = 0;
        std::uint64_t reserved = 0;
        if (!reader.u(layout.flags) || !reader.u(count) || !reader.u(reserved))
            return fail<ContainerLayout>(Errc::corrupt_index, "header");
        if (count > kMaxSections ||
            count > (bytes.size() - header_size - trailer_size) / entry_size)
            return fail<ContainerLayout>(Errc::corrupt_index, "invalid section count");

        layout.sections.resize(count);
        std::set<std::uint32_t> tags;
        for (auto& section : layout.sections) {
            if (!reader.u(section.tag) || !reader.u(section.offset) || !reader.u(section.length))
                return fail<ContainerLayout>(Errc::corrupt_index, "section table");
            if (!tags.insert(section.tag).second)
                return fail<ContainerLayout>(Errc::corrupt_index, "duplicate section");
        }

        const std::uint64_t payload_begin = header_size + entry_size * count;
        const std::uint64_t payload_end = bytes.size() - trailer_size;
        std::sort(layout.sections.begin(), layout.sections.end(),
                  [](const SectionRange& left, const SectionRange& right) {
                      return left.offset < right.offset;
                  });
        std::uint64_t previous_end = payload_begin;
        for (const auto& section : layout.sections) {
            if (section.offset < payload_begin || section.offset > payload_end ||
                section.length > payload_end - section.offset)
                return fail<ContainerLayout>(Errc::corrupt_index, "section out of bounds");
            if (section.offset < previous_end)
                return fail<ContainerLayout>(Errc::corrupt_index, "overlapping sections");
            previous_end = section.offset + section.length;
        }
        return layout;
    } catch (const std::bad_alloc&) {
        return fail<ContainerLayout>(Errc::corrupt_index, "container allocation failed");
    } catch (const std::exception& error) {
        return fail<ContainerLayout>(Errc::corrupt_index,
                                     std::string("container parse failed: ") + error.what());
    } catch (...) {
        return fail<ContainerLayout>(Errc::corrupt_index, "container parse failed");
    }
}

} // namespace rag::store
