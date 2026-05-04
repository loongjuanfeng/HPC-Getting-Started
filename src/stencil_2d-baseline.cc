#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <print>
#include <random>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <vector>

#include <CLI/CLI.hpp>

namespace config {
constexpr std::size_t default_grid_size = 8192;
constexpr std::size_t default_iterations = 100;
} // namespace config

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
} timer;

int main(int argc, char* argv[]) {
    CLI::App app{"2D stencil baseline template"};
    std::size_t grid_size = config::default_grid_size;
    std::size_t iterations = config::default_iterations;

    app.add_option("-s,--size", grid_size, "Grid dimension (N x N)");
    app.add_option("-i,--iterations", iterations, "Stencil iteration count");
    CLI11_PARSE(app, argc, argv);

    auto report_logger = spdlog::stdout_color_mt("REPORT");

    // Instructions:
    // 1. Allocate two N x N float grids: current and next.
    // 2. Initialize current with a known heat source pattern.
    // 3. For each iteration, update interior cells only: rows [1, N - 2], cols [1, N - 2].
    // 4. Use the 5-point stencil:
    //      next[i][j] = 0.25F * (current[i - 1][j] + current[i + 1][j]
    //                         + current[i][j - 1] + current[i][j + 1]);
    // 5. Swap current and next after every iteration.
    // 6. Measure only the iteration loop, not allocation or initialization.
    // 7. Report time, GUpdates/s, GFLOPS, effective GB/s, and checksum.
    // 8. Add a checksum so compiler optimizations cannot delete the work.

    using data_type = float;

    // initialization
    std::vector<data_type> current(grid_size * grid_size);
    std::vector<data_type> next(grid_size * grid_size);

    // random generation
    std::mt19937_64 random_number_generator(std::random_device{}());
    std::normal_distribution<data_type> normal_distribution{1.0};
    for (std::size_t i = 0; i < grid_size * grid_size; i++) {
        current[i] = (normal_distribution(random_number_generator) > 1.0 ? 1.0F : 0.0F);
    }

    timer.start();
    // iteration
    for (std::size_t iteration = 0; iteration < iterations; iteration++) {
        std::print("\r\033[2K[INFO] Iteration: {:>4} / {:<4}", iteration + 1, iterations);
        std::cout << std::flush;

        for (std::size_t i = 1; i + 1 < grid_size; i++) {
            for (std::size_t j = 1; j + 1 < grid_size; j++) {
                next[(i * grid_size) + j] =
                    0.25F *
                    (current[((i - 1) * grid_size) + j] + current[((i + 1) * grid_size) + j] +
                     current[(i * grid_size) + (j - 1)] + current[(i * grid_size) + (j + 1)]);
            }
        }

        std::swap(current, next);
    }
    std::cout << std::endl;
    timer.end();

    double checksum = 0.0;
#pragma omp parallel for reduction(+ : checksum)
    for (auto value : current) {
        checksum += value;
    }

    constexpr double giga_bytes = 1e9;
    const auto updates = static_cast<double>(iterations) * static_cast<double>(grid_size - 2) *
                         static_cast<double>(grid_size - 2);
    const double flops = updates * 4.0;
    const double effective_bytes = updates * 5.0 * sizeof(data_type);
    const double seconds = timer.elapsed();
    report_logger->info("=== 2D Stencil Baseline Template ===");
    report_logger->info("{:>12} = {:>10} x {:<10}", "grid", grid_size, grid_size);
    report_logger->info("{:>12} = {:>10}", "iterations", iterations);
    report_logger->info("{:>12} = {:>10.4f} s", "time", seconds);
    report_logger->info("{:>12} = {:>10.4f} GUpdates/s", "updates", updates / seconds / giga_bytes);
    report_logger->info("{:>12} = {:>10.4f} GFLOPS", "compute", flops / seconds / giga_bytes);
    report_logger->info("{:>12} = {:>10.4f} GB/s", "bandwidth",
                        effective_bytes / seconds / giga_bytes);
    report_logger->info("{:>12} = {:>10.4f}", "checksum",
                        checksum / static_cast<double>(grid_size * grid_size));

    return EXIT_SUCCESS;
}
