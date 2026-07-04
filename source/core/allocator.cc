#include "allocator.hh"

#include <cstdlib>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace core {
namespace {
[[nodiscard]] std::size_t page_aligned_size(const std::size_t bytes) {
        if (bytes == 0) {
                return 0;
        }

#if defined(__unix__) || defined(__APPLE__)
        const auto page_size = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
        const auto remainder = bytes % page_size;
        if (remainder == 0) {
                return bytes;
        }
        return bytes + page_size - remainder;
#else
        return bytes;
#endif
}

[[nodiscard]] std::size_t system_page_size() {
#if defined(__unix__) || defined(__APPLE__)
        return static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
#else
        return alignof(std::max_align_t);
#endif
}
}  // namespace

void* allocate_bytes_without_initialization(
    const std::size_t bytes, const std::size_t alignment) {
        if (bytes == 0) {
                return nullptr;
        }

        if (alignment > system_page_size()) {
                return ::operator new(bytes, std::align_val_t{alignment});
        }

#if defined(__unix__) || defined(__APPLE__)
        const auto mapped_bytes = page_aligned_size(bytes);

#if defined(MAP_HUGETLB)
        void* pointer = ::mmap(
            nullptr, mapped_bytes, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (pointer == MAP_FAILED) {
                pointer = ::mmap(
                    nullptr, mapped_bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        }
#else
        void* pointer = ::mmap(
            nullptr, mapped_bytes, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif

        if (pointer == MAP_FAILED) {
                throw std::bad_alloc();
        }

#if defined(MADV_HUGEPAGE)
        ::madvise(pointer, mapped_bytes, MADV_HUGEPAGE);
#endif

        return pointer;
#else
        void* pointer = nullptr;
        if (::posix_memalign(&pointer, alignment, bytes) != 0) {
                throw std::bad_alloc();
        }
        return pointer;
#endif
}

void deallocate_bytes_without_initialization(
    void* pointer, const std::size_t bytes, const std::size_t alignment) noexcept {
        if (pointer == nullptr) {
                return;
        }

        if (alignment > system_page_size()) {
                ::operator delete(pointer, bytes, std::align_val_t{alignment});
                return;
        }

#if defined(__unix__) || defined(__APPLE__)
        ::munmap(pointer, page_aligned_size(bytes));
#else
        std::free(pointer);
#endif
}
}  // namespace core
