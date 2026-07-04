#include <chrono>
#include <cstddef>
#include <new>
#include <sys/mman.h>
#include <type_traits>
#include <vector>

#include "omp.h"

#include <spdlog/spdlog.h>

namespace config {
constexpr bool use_no_initialization_allocator = true;
constexpr std::size_t default_matrix_size = 4096; // N times N
constexpr double checksum_tolerance = 500.0;
constexpr std::size_t calculation_loop = 5;
} // namespace config

template <class Type> struct no_initialization_allocator {
    using value_type = Type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::true_type;
    explicit no_initialization_allocator() noexcept = default;

    [[nodiscard]] Type* allocate(std::size_t size) {
        if (size > std::numeric_limits<std::size_t>::max() / sizeof(Type)) {
            throw std::bad_array_new_length();
        }

        const auto bytes = size * sizeof(Type);
        // void* pointer = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
        //                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        void* pointer =
            ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (pointer == MAP_FAILED) {
            throw std::bad_alloc();
        }

        ::madvise(pointer, bytes, MADV_HUGEPAGE);
        return static_cast<Type*>(pointer);
    }

    void deallocate(Type* pointer, std::size_t size) noexcept {
        ::munmap(pointer, size * sizeof(Type));
    }

    [[nodiscard]] constexpr std::size_t max_size() const noexcept {
        return std::numeric_limits<std::size_t>::max() / sizeof(Type);
    }

    template <class U>
    void construct(U* pointer) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(pointer)) U;
    }

    template <class U, class... Args>
        requires(sizeof...(Args) > 0)
    void construct(U* pointer,
                   Args&&... args) noexcept(std::is_nothrow_constructible_v<U, Args...>) {
        ::new (static_cast<void*>(pointer)) U(std::forward<Args>(args)...);
    }

    template <class U> void destroy(U* pointer) noexcept { std::destroy_at(pointer); }

    template <class U> bool operator==(const no_initialization_allocator<U>&) const noexcept {
        return true;
    }
};

template <typename Type, std::size_t matrix_count> auto create_matrix(std::size_t matrix_size) {
    using vector_type =
        std::conditional_t<config::use_no_initialization_allocator,
                           std::vector<Type, no_initialization_allocator<Type>>, std::vector<Type>>;
    std::array<vector_type, matrix_count> matrices;
    for (auto& matrix : matrices) {
        matrix = vector_type(matrix_size * matrix_size);
    }
    return matrices;
}

class Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point time_start;
    Clock::time_point time_end;

public:
    void start() { time_start = Clock::now(); }
    void end() { time_end = Clock::now(); }

    [[nodiscard]] auto elapsed() const {
        return std::chrono::duration<double>(time_end - time_start).count();
    }
};
