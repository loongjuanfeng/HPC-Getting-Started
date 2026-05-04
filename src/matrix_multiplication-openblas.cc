#include "cblas.h"
#include "spdlog/common.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <random>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <type_traits>
#include <vector>

#include <CLI/CLI.hpp>

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
        void* pointer = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
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

int main(int argc, char* argv[]) {
    CLI::App app{"OpenBLAS matrix multiplication - HPC"};
    std::size_t matrix_size{config::default_matrix_size};
    app.add_option("-s,--size", matrix_size, "Matrix dimension (N x N)");
    CLI11_PARSE(app, argc, argv);

    auto [A, B, C] = create_matrix<float, 3>(matrix_size);

    std::mt19937 random_number_generator(std::random_device{}());
    std::normal_distribution<float> normal_distribution{1.0};
    for (std::size_t i = 0; i < matrix_size * matrix_size; i++) {
        A[i] = normal_distribution(random_number_generator);
        B[i] = normal_distribution(random_number_generator);
    }

    auto report_logger = spdlog::stdout_color_mt("REPORT");
    Timer timer;
    double best_time = std::numeric_limits<double>::max();

    for (std::size_t loop_index = 0; loop_index < config::calculation_loop; loop_index++) {
        for (std::size_t i = 0; i < matrix_size * matrix_size; i++) {
            C[i] = .0;
        }

        timer.start();

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, matrix_size, matrix_size,
                    matrix_size, 1.0F, A.data(), matrix_size, B.data(), matrix_size, 1.0F,
                    C.data(), matrix_size);

        timer.end();

        best_time = std::min(best_time, timer.elapsed());
        spdlog::info("loop {}, time: {}", loop_index, timer.elapsed());
    }

    float expected = 0.0F;
    for (std::size_t k = 0; k < matrix_size; k++) {
        expected += A[k] * B[k * matrix_size];
    }
    if (std::abs(C[0] - expected) > config::checksum_tolerance) {
        spdlog::error("BLAS Verification failed: C[0] = {} expected {}", C[0], expected);
        return EXIT_FAILURE;
    }

    const double giga_bytes = 1e9;
    const double total_data_size =
        3.0 * static_cast<double>(matrix_size * matrix_size * sizeof(float)) / giga_bytes;
    const double flops = 2.0 * static_cast<double>(matrix_size) *
                         static_cast<double>(matrix_size) * static_cast<double>(matrix_size);
    report_logger->info("=== OpenBLAS sgemm ===");
    report_logger->info("{:>12} = {:>10.0f} x {:<10.0f}", "size",
                        static_cast<double>(matrix_size), static_cast<double>(matrix_size));
    report_logger->info("{:>12} = {:>10.4f} s", "time used", best_time);
    report_logger->info("{:>12} = {:>10.4f} GFLOPS", "compute", flops / best_time / giga_bytes);
    report_logger->info("{:>12} = {:>10.4f} GB/s", "bandwidth", total_data_size / best_time);

    return EXIT_SUCCESS;
}
