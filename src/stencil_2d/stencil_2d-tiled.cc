#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <print>
#include <random>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <thread>
#include <vector>

#include <CLI/CLI.hpp>

namespace config {
constexpr std::size_t default_grid_size = 8192;
constexpr std::size_t default_iterations = 100;
constexpr std::size_t default_tile_size = 256;
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

void setup_threads();

int main(int argc, char* argv[]) {
    CLI::App app{"2D stencil tiled template"};
    std::size_t grid_size = config::default_grid_size;
    std::size_t iterations = config::default_iterations;
    std::size_t tile_size = config::default_tile_size;
    bool show_progress = false;

    app.add_option("-s,--size", grid_size, "Grid dimension (N x N)");
    app.add_option("-i,--iterations", iterations, "Stencil iteration count");
    app.add_option("-t,--tile", tile_size, "Tile dimension");
    app.add_flag("-p,--show-progress", show_progress, "Show the iteration progress");
    CLI11_PARSE(app, argc, argv);

    if (grid_size < 3) {
        spdlog::error("grid size must be at least 3");
        return EXIT_FAILURE;
    }
    if (tile_size == 0) {
        spdlog::error("tile size must be greater than 0");
        return EXIT_FAILURE;
    }

    auto report_logger = spdlog::stdout_color_mt("REPORT");
    setup_threads();

    using data_type = float;

    std::vector<data_type> current(grid_size * grid_size);
    std::vector<data_type> next(grid_size * grid_size);

    // Instructions:
    // 1. Start from the baseline version after it is correct.
    // 2. Split the interior domain into row and column tiles.
    // 3. For each tile, clamp tile end indices so boundary cells are not updated.
    // 4. Keep the update formula identical to baseline.
    // 5. Test tile sizes such as 32, 64, 128, 256, and 512.
    // 6. Compare against baseline with the same checksum.
    // 7. Add OpenMP only after the single-thread tiled version is correct.
    // 8. Report tile size with time, GUpdates/s, GFLOPS, effective GB/s, and checksum.

    {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < grid_size * grid_size; i++) {
            thread_local std::mt19937_64 random_number_generator(std::random_device{}());
            thread_local std::normal_distribution<data_type> normal_distribution{1.0};
            current[i] = (normal_distribution(random_number_generator) > 1.0 ? 1.0F : 0.0F);
        }

        timer.start();
// iteration
#pragma omp parallel
        for (std::size_t iteration = 0; iteration < iterations; iteration++) {
            if (show_progress) {
#pragma omp single
                {
                std::print("\r\033[2K[INFO] Iteration: {:>4} / {:<4}", iteration + 1, iterations);
                std::cout << std::flush;
                }
            }

#pragma omp for schedule(static)
            for (std::size_t ii = 1; ii < grid_size - 1; ii += tile_size) {
                const auto i_end = std::min(ii + tile_size, grid_size - 1);

                for (std::size_t i = ii; i < i_end; i++) {
                    const data_type* const previous_row = current.data() + ((i - 1) * grid_size);
                    const data_type* const current_row = current.data() + (i * grid_size);
                    const data_type* const next_row = current.data() + ((i + 1) * grid_size);
                    data_type* const out = next.data() + (i * grid_size);

                    for (std::size_t jj = 1; jj < grid_size - 1; jj += tile_size) {
                        const auto j_end = std::min(jj + tile_size, grid_size - 1);
#pragma omp simd
                        for (std::size_t j = jj; j < j_end; j++) {
                            out[j] = 0.25F * (previous_row[j] + current_row[j - 1] +
                                              current_row[j + 1] + next_row[j]);
                        }
                    }
                }
            }
#pragma omp single
            std::swap(current, next);

        }
        if (show_progress) {
            std::cout << std::endl;
        }
        timer.end();
    }

    double checksum = 0.0;
#pragma omp parallel for reduction(+ : checksum)
    for (auto value : current) {
        checksum += value;
    }

    constexpr double giga_bytes = 1e9;
    const double seconds = timer.elapsed();
    const auto updates = static_cast<double>(iterations) * static_cast<double>(grid_size - 2) *
                         static_cast<double>(grid_size - 2);
    const double flops = updates * 4.0;
    const double effective_bytes = updates * 5.0 * sizeof(data_type);

    report_logger->info("=== 2D Stencil Tiled Template ===");
    report_logger->info("{:>12} = {:>10} x {:<10}", "grid", grid_size, grid_size);
    report_logger->info("{:>12} = {:>10}", "iterations", iterations);
    report_logger->info("{:>12} = {:>10}", "tile", tile_size);
    report_logger->info("{:>12} = {:>10.4f} s", "time", seconds);
    report_logger->info("{:>12} = {:>10.4f} GUpdates/s", "updates", updates / seconds / giga_bytes);
    report_logger->info("{:>12} = {:>10.4f} GFLOPS", "compute", flops / seconds / giga_bytes);
    report_logger->info("{:>12} = {:>10.4f} GB/s", "bandwidth",
                        effective_bytes / seconds / giga_bytes);
    report_logger->info("{:>12} = {:>10.4f}", "checksum", checksum);

    return EXIT_SUCCESS;
}

void setup_threads() {
    const auto* const omp_num_threads = std::getenv("OMP_NUM_THREADS");
    const auto threads_count = static_cast<bool>(omp_num_threads)
                                   ? std::atoi(omp_num_threads)
                                   : static_cast<int>(std::thread::hardware_concurrency());
    omp_set_num_threads(threads_count);
    spdlog::info("threads = {}", threads_count);
}
