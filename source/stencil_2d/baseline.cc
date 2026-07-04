#include <CLI/CLI.hpp>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>

#include "header.hh"
#include "log.hh"
#include "timer.hh"

namespace {
void initialize_grids(grid_vector<float>& current, grid_vector<float>& next,
                      const std::size_t grid_size) {
        for (std::size_t i = 0; i < grid_size; i++) {
                for (std::size_t j = 0; j < grid_size; j++) {
                        const auto index = (i * grid_size) + j;
                        current[index] = heat_source_value(i, j, grid_size);
                        next[index] = 0.0F;
                }
        }
}

void stencil_step(const grid_vector<float>& current, grid_vector<float>& next,
                  const std::size_t grid_size) {
        for (std::size_t i = 1; i < grid_size - 1; i++) {
                const float* const previous_row =
                    current.data() + ((i - 1) * grid_size);
                const float* const current_row =
                    current.data() + (i * grid_size);
                const float* const next_row =
                    current.data() + ((i + 1) * grid_size);
                float* const output = next.data() + (i * grid_size);

                for (std::size_t j = 1; j < grid_size - 1; j++) {
                        output[j] =
                            0.25F * (previous_row[j] + next_row[j] +
                                     current_row[j - 1] + current_row[j + 1]);
                }
        }
}

void run_stencil(grid_vector<float>& current, grid_vector<float>& next,
                 const std::size_t grid_size, const std::size_t iterations) {
        for (std::size_t iteration = 0; iteration < iterations; iteration++) {
                stencil_step(current, next, grid_size);
                std::swap(current, next);
        }
}
}  // namespace

int main(int argc, char* argv[]) {
        CLI::App app{"2D stencil baseline - HPC"};
        std::size_t grid_size{config::default_grid_size};
        std::size_t iterations{config::default_iterations};
        app.add_option("-s,--size", grid_size, "Grid dimension (N x N)")
            ->check(CLI::Range(std::size_t{3},
                               std::numeric_limits<std::size_t>::max()));
        app.add_option("-i,--iterations", iterations, "Stencil iteration count")
            ->check(CLI::PositiveNumber);
        CLI11_PARSE(app, argc, argv);

        auto [current, next] = create_grids<float, 2>(grid_size);
        initialize_grids(current, next, grid_size);

        core::Timer timer;
        timer.start();
        run_stencil(current, next, grid_size, iterations);
        timer.end();

        const auto seconds = timer.elapsed();
        const auto checksum = checksum_grid(current);
        if (!std::isfinite(checksum)) {
                core::ERROR("checksum failed: {}", checksum);
                return EXIT_FAILURE;
        }

        const auto metrics = calculate_metrics(grid_size, iterations);
        std::cout << core::write(stencil_report{
                         .mode = "baseline",
                         .grid_size = grid_size,
                         .iterations = iterations,
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
