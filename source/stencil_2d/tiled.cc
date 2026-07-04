#include <omp.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <thread>
#include <utility>

#include "header.hh"
#include "log.hh"
#include "timer.hh"

namespace {
void setup_threads() {
        const auto* const omp_num_threads = std::getenv("OMP_NUM_THREADS");
        auto threads_count =
            static_cast<bool>(omp_num_threads)
                ? std::atoi(omp_num_threads)
                : static_cast<int>(std::thread::hardware_concurrency());
        threads_count = std::max(threads_count, 1);
        omp_set_dynamic(0);
        omp_set_num_threads(threads_count);
        core::INFO("threads = {}", threads_count);
}

void initialize_grids(grid_vector<float>& current, grid_vector<float>& next,
                      const std::size_t grid_size) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < grid_size; i++) {
                for (std::size_t j = 0; j < grid_size; j++) {
                        const auto index = (i * grid_size) + j;
                        current[index] = heat_source_value(i, j, grid_size);
                        next[index] = 0.0F;
                }
        }
}

void run_stencil(grid_vector<float>& current, grid_vector<float>& next,
                 const std::size_t grid_size, const std::size_t iterations,
                 const std::size_t tile_size) {
#pragma omp parallel
        {
                for (std::size_t iteration = 0; iteration < iterations;
                     iteration++) {
#pragma omp for schedule(static) collapse(2)
                        for (std::size_t ii = 1; ii < grid_size - 1;
                             ii += tile_size) {
                                for (std::size_t jj = 1; jj < grid_size - 1;
                                     jj += tile_size) {
                                        const auto i_end = std::min(
                                            ii + tile_size, grid_size - 1);
                                        const auto j_end = std::min(
                                            jj + tile_size, grid_size - 1);

                                        for (std::size_t i = ii; i < i_end;
                                             i++) {
                                                const float* const
                                                    previous_row =
                                                        current.data() +
                                                        ((i - 1) * grid_size);
                                                const float* const current_row =
                                                    current.data() +
                                                    (i * grid_size);
                                                const float* const next_row =
                                                    current.data() +
                                                    ((i + 1) * grid_size);
                                                float* const output =
                                                    next.data() +
                                                    (i * grid_size);

#pragma omp simd
                                                for (std::size_t j = jj;
                                                     j < j_end; j++) {
                                                        output[j] =
                                                            0.25F *
                                                            (previous_row[j] +
                                                             next_row[j] +
                                                             current_row[j -
                                                                         1] +
                                                             current_row[j +
                                                                         1]);
                                                }
                                        }
                                }
                        }

#pragma omp single
                        std::swap(current, next);
                }
        }
}
}  // namespace

int main(int argc, char* argv[]) {
        CLI::App app{"2D stencil tiled OpenMP - HPC"};
        std::size_t grid_size{config::default_grid_size};
        std::size_t iterations{config::default_iterations};
        std::size_t tile_size{config::default_tile_size};
        app.add_option("-s,--size", grid_size, "Grid dimension (N x N)")
            ->check(CLI::Range(std::size_t{3},
                               std::numeric_limits<std::size_t>::max()));
        app.add_option("-i,--iterations", iterations, "Stencil iteration count")
            ->check(CLI::PositiveNumber);
        app.add_option("-t,--tile", tile_size, "Tile dimension")
            ->check(CLI::PositiveNumber);
        CLI11_PARSE(app, argc, argv);

        setup_threads();

        auto [current, next] = create_grids<float, 2>(grid_size);
        initialize_grids(current, next, grid_size);

        core::Timer timer;
        timer.start();
        run_stencil(current, next, grid_size, iterations, tile_size);
        timer.end();

        const auto seconds = timer.elapsed();
        const auto checksum = checksum_grid(current);
        if (!std::isfinite(checksum)) {
                core::ERROR("checksum failed: {}", checksum);
                return EXIT_FAILURE;
        }

        const auto metrics = calculate_metrics(grid_size, iterations);
        std::cout << core::write(tiled_stencil_report{
                         .mode = "tiled",
                         .grid_size = grid_size,
                         .iterations = iterations,
                         .tile_size = tile_size,
                         .threads = omp_get_max_threads(),
                         .seconds = seconds,
                         .gupdates = metrics.updates / seconds / 1e9,
                         .gflops = metrics.flops / seconds / 1e9,
                         .bandwidth_gbs =
                             metrics.effective_data_size_gb / seconds,
                         .checksum = checksum,
                     })
                  << '\n';

        return EXIT_SUCCESS;
}
