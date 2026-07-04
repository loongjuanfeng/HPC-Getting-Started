#include <omp.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <vector>

#include "header.hh"
#include "log.hh"
#include "timer.hh"

namespace config {
constexpr std::size_t tiles_size[] = {128, 256, 512};
}  // namespace config

namespace {
void setup_threads() {
        const auto* const omp_num_threads = std::getenv("OMP_NUM_THREADS");
        const auto threads_count =
            static_cast<bool>(omp_num_threads)
                ? std::atoi(omp_num_threads)
                : static_cast<int>(std::thread::hardware_concurrency());
        omp_set_num_threads(threads_count);
        core::INFO("threads = {}", threads_count);
}

void initialize_inputs(matrix_vector<float>& A, matrix_vector<float>& B,
                       const std::size_t matrix_size) {
#pragma omp parallel for
        for (std::size_t i = 0; i < matrix_size * matrix_size; i++) {
                thread_local std::mt19937 random_number_generator(
                    std::random_device{}());
                thread_local std::normal_distribution<float>
                    normal_distribution{1.0};
                A[i] = normal_distribution(random_number_generator);
                B[i] = normal_distribution(random_number_generator);
        }
}

void zero_matrix(matrix_vector<float>& matrix, const std::size_t matrix_size) {
#pragma omp parallel for
        for (std::size_t i = 0; i < matrix_size * matrix_size; i++) {
                matrix[i] = 0.0F;
        }
}

void multiply_tiled(const matrix_vector<float>& A,
                    const matrix_vector<float>& B, matrix_vector<float>& C,
                    const std::size_t matrix_size,
                    const std::size_t tile_size) {
#pragma omp parallel
        {
                std::vector<float> B_packed(tile_size * tile_size);

#pragma omp for
                for (std::size_t ii = 0; ii < matrix_size; ii += tile_size) {
                        for (std::size_t kk = 0; kk < matrix_size;
                             kk += tile_size) {
                                for (std::size_t jj = 0; jj < matrix_size;
                                     jj += tile_size) {
                                        const auto i_end = std::min(
                                            ii + tile_size, matrix_size);
                                        const auto k_end = std::min(
                                            kk + tile_size, matrix_size);
                                        const auto j_end = std::min(
                                            jj + tile_size, matrix_size);

                                        for (std::size_t k = kk; k < k_end;
                                             k++) {
                                                for (std::size_t j = jj;
                                                     j < j_end; j++) {
                                                        B_packed[((k - kk) *
                                                                  tile_size) +
                                                                 (j - jj)] =
                                                            B[(k *
                                                               matrix_size) +
                                                              j];
                                                }
                                        }

                                        for (std::size_t i = ii; i < i_end;
                                             i++) {
                                                for (std::size_t k = kk;
                                                     k < k_end; k++) {
                                                        const float A_ik =
                                                            A[(i *
                                                               matrix_size) +
                                                              k];
#pragma omp simd
                                                        for (std::size_t j = jj;
                                                             j < j_end; j++) {
                                                                C[(i *
                                                                   matrix_size) +
                                                                  j] +=
                                                                    A_ik *
                                                                    B_packed
                                                                        [((k -
                                                                           kk) *
                                                                          tile_size) +
                                                                         (j -
                                                                          jj)];
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }
}

bool verify_first_element(const matrix_vector<float>& A,
                          const matrix_vector<float>& B,
                          const matrix_vector<float>& C,
                          const std::size_t matrix_size) {
        float expected = 0.0F;
        for (std::size_t k = 0; k < matrix_size; k++) {
                expected += A[k] * B[k * matrix_size];
        }

        if (std::abs(C[0] - expected) <= config::checksum_tolerance) {
                return true;
        }

        core::ERROR("Verification failed: C[0] = {} expected {}", C[0],
                    expected);
        return false;
}
}  // namespace

int main(int argc, char* argv[]) {
        CLI::App app{"hand-written matrix multiplication - HPC"};
        std::size_t matrix_size{config::default_matrix_size};
        app.add_option("-s,--size", matrix_size, "Matrix dimension (N x N)");
        CLI11_PARSE(app, argc, argv);

        setup_threads();

        auto [A, B, C] = create_matrix<float, 3>(matrix_size);
        initialize_inputs(A, B, matrix_size);

        core::Timer timer;

        for (const auto tile_size : config::tiles_size) {
                core::INFO("tile size {}", tile_size);

                double best_time = std::numeric_limits<double>::max();

                for (std::size_t loop_index = 0;
                     loop_index < config::calculation_loop; loop_index++) {
                        zero_matrix(C, matrix_size);

                        timer.start();
                        multiply_tiled(A, B, C, matrix_size, tile_size);
                        timer.end();

                        best_time = std::min(best_time, timer.elapsed());
                        core::INFO("loop {}, time: {}", loop_index,
                                   timer.elapsed());
                }

                if (!verify_first_element(A, B, C, matrix_size)) {
                        return EXIT_FAILURE;
                }

                const auto metrics = calculate_metrics(matrix_size);
                std::cout << core::write(tiled_matrix_report{
                                 .mode = "hand-written",
                                 .matrix_size = matrix_size,
                                 .tile_size = tile_size,
                                 .seconds = best_time,
                                 .gflops = metrics.flops / best_time / 1e9,
                                 .bandwidth_gbs =
                                     metrics.total_data_size_gb / best_time,
                             })
                          << '\n';
        }

        return EXIT_SUCCESS;
}
