#include "spdlog/common.h"
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <omp.h>
#include <random>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <thread>
#include <type_traits>
#include <vector>

#include <CLI/CLI.hpp>

namespace config {
constexpr bool use_no_initilization_allocator = true;
constexpr std::size_t default_vector_size = 2'000'000'000;
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

void setup_threads();

template <typename Type, std::size_t vector_count> auto create_vectors(std::size_t vector_size) {
    using vector_type =
        std::conditional_t<config::use_no_initilization_allocator,
                           std::vector<Type, no_initialization_allocator<Type>>, std::vector<Type>>;
    std::array<vector_type, vector_count> vectors;
    for (auto& vector : vectors) {
        vector = vector_type(vector_size);
    }
    return vectors;
}

class Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point time_start;
    Clock::time_point time_end;

public:
    void start();
    void end();

    [[nodiscard]] auto elapsed() const {
        return std::chrono::duration<double>(time_end - time_start).count();
    }
};
Timer timer;

int main(int argc, char* argv[]) {
    CLI::App app{"vector addition - HPC starter"};
    std::size_t vector_size{config::default_vector_size};
    app.add_option("-s,--size", vector_size, "Number of elements");
    CLI11_PARSE(app, argc, argv);

    setup_threads();

    auto [vector_1, vector_2, sum_vector] = create_vectors<float, 3>(vector_size);

#pragma omp parallel for
    for (std::size_t i = 0; i < vector_size; i++) {
        thread_local std::mt19937 random_number_generator(std::random_device{}());
        thread_local std::normal_distribution<float> normal_distribution{1.0};
        vector_1[i] = normal_distribution(random_number_generator);
        vector_2[i] = normal_distribution(random_number_generator);
    }

    timer.start();

#pragma omp parallel for
    for (std::size_t i = 0; i < vector_size; i++) {
        sum_vector[i] = vector_1[i] * vector_2[i];
    }

    timer.end();

    double vector_sum{0.0};
#pragma omp parallel for reduction(+ : vector_sum)
    for (std::size_t i = 0; i < vector_size; i++) {
        vector_sum += sum_vector[i];
    }

    auto report_logger = spdlog::stdout_color_mt("REPORT");

    const double vector_total_size = 3.0 * static_cast<double>(vector_size * sizeof(float)) / 1e9;
    report_logger->info("=== Vector Addition ===");
    report_logger->info("{:>12} = {:>10.0f}", "size", static_cast<double>(vector_size));
    report_logger->info("{:>12} = {:>10.4f} s", "time used", timer.elapsed());
    report_logger->info("{:>12} = {:>10.4f} GB/s", "bandwidth",
                        vector_total_size / timer.elapsed());
    report_logger->info("{:>12} = {:>10.4f}", "mean", vector_sum / vector_size);

    return EXIT_SUCCESS;
}

// helper functions

void setup_threads() {
    const auto* const OMP_NUM_THREADS = std::getenv("OMP_NUM_THREADS");
    const auto threads_count = static_cast<bool>(OMP_NUM_THREADS)
                                   ? std::atoi(OMP_NUM_THREADS)
                                   : static_cast<int>(std::thread::hardware_concurrency());
    omp_set_num_threads(threads_count);
    spdlog::info("threads = {}", threads_count);
}


void Timer::start() {
    time_start = Clock::now();
}
void Timer::end() {
    time_end = Clock::now();
}
