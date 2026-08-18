#include "rag/store/container_view.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <exception>
#include <limits>
#include <new>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rag::store {

ContainerView::~ContainerView() { reset(); }

ContainerView::ContainerView(ContainerView&& other) noexcept
    : data_(other.data_), size_(other.size_), sections_(std::move(other.sections_)),
      flags_(other.flags_), major_(other.major_), minor_(other.minor_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

ContainerView& ContainerView::operator=(ContainerView&& other) noexcept {
    if (this == &other)
        return *this;
    reset();
    data_ = other.data_;
    size_ = other.size_;
    sections_ = std::move(other.sections_);
    flags_ = other.flags_;
    major_ = other.major_;
    minor_ = other.minor_;
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
}

void ContainerView::reset() noexcept {
    if (data_ != nullptr)
        ::munmap(const_cast<char*>(data_), size_);
    data_ = nullptr;
    size_ = 0;
    sections_.clear();
}

Result<ContainerView> ContainerView::open_file(const std::string& path) {
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0)
        return fail<ContainerView>(Errc::io_error, "open " + path + ": " + std::strerror(errno));

    struct stat status{};
    if (::fstat(descriptor, &status) != 0) {
        const std::string message = std::strerror(errno);
        ::close(descriptor);
        return fail<ContainerView>(Errc::io_error, "stat " + path + ": " + message);
    }
    if (status.st_size <= 0 ||
        static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
        ::close(descriptor);
        return fail<ContainerView>(Errc::corrupt_index, "invalid container file size");
    }

    const auto size = static_cast<std::size_t>(status.st_size);
    void* mapped = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, descriptor, 0);
    const int map_error = errno;
    ::close(descriptor);
    if (mapped == MAP_FAILED)
        return fail<ContainerView>(Errc::io_error, "map " + path + ": " + std::strerror(map_error));

    try {
        ContainerView view;
        view.data_ = static_cast<const char*>(mapped);
        view.size_ = size;
        auto layout = validate_container_layout(std::string_view(view.data_, view.size_));
        if (!layout)
            return unexpected(layout.error());
        view.flags_ = layout->flags;
        view.major_ = layout->format_major;
        view.minor_ = layout->format_minor;
        view.sections_ = std::move(layout->sections);
        return view;
    } catch (const std::bad_alloc&) {
        return fail<ContainerView>(Errc::corrupt_index, "container view allocation failed");
    } catch (const std::exception& error) {
        return fail<ContainerView>(Errc::io_error,
                                   "map initialization failed: " + std::string(error.what()));
    }
}

bool ContainerView::has(Tag tag) const { return get(tag).has_value(); }

std::optional<std::string_view> ContainerView::get(Tag tag) const {
    return get_raw(static_cast<std::uint32_t>(tag));
}

std::optional<std::string_view> ContainerView::get_raw(std::uint32_t tag) const {
    const auto found =
        std::find_if(sections_.begin(), sections_.end(),
                     [tag](const SectionRange& section) { return section.tag == tag; });
    if (found == sections_.end())
        return std::nullopt;
    return std::string_view(data_ + found->offset, static_cast<std::size_t>(found->length));
}

} // namespace rag::store
