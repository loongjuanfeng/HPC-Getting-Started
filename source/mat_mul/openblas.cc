#include <openblas/cblas.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <thread>

#include "header.hh"
#include "log.hh"
#include "timer.hh"

namespace {
void setup_threads() {
        const auto* const openblas_num_threads =
            std::getenv("OPENBLAS_NUM_THREADS");
        const auto threads_count =
            static_cast<bool>(openblas_num_threads)
                ? std::atoi(openblas_num_threads)
                : static_cast<int>(std::thread::hardware_concurrency());
        openblas_set_num_threads(threads_count);
        core::INFO("threads = {}", threads_count);
}
}  // namespace

int main(int argc, char* argv[]) {
        CLI::App app{"OpenBLAS matrix multiplication - HPC"};
        std::size_t matrix_size{config::default_matrix_size};
        app.add_option("-s,--size", matrix_size, "Matrix dimension (N x N)");
        CLI11_PARSE(app, argc, argv);

        setup_threads();

        auto [A, B, C] = create_matrix<float, 3>(matrix_size);

        std::mt19937 random_number_generator(std::random_device{}());
        std::normal_distribution<float> normal_distribution{1.0};
        for (std::size_t i = 0; i < matrix_size * matrix_size; i++) {
                A[i] = normal_distribution(random_number_generator);
                B[i] = normal_distribution(random_number_generator);
        }

        core::Timer timer;
        double best_time = std::numeric_limits<double>::max();

        for (std::size_t loop_index = 0; loop_index < config::calculation_loop;
             loop_index++) {
                for (std::size_t i = 0; i < matrix_size * matrix_size; i++) {
                        C[i] = 0.0F;
                }

                timer.start();

                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(matrix_size),
                            static_cast<int>(matrix_size),
                            static_cast<int>(matrix_size), 1.0F, A.data(),
                            static_cast<int>(matrix_size), B.data(),
                            static_cast<int>(matrix_size), 1.0F, C.data(),
                            static_cast<int>(matrix_size));

                timer.end();

                best_time = std::min(best_time, timer.elapsed());
                core::INFO("loop {}, time: {}", loop_index, timer.elapsed());
        }

        float expected = 0.0F;
        for (std::size_t k = 0; k < matrix_size; k++) {
                expected += A[k] * B[k * matrix_size];
        }
        if (std::abs(C[0] - expected) > config::checksum_tolerance) {
                core::ERROR("BLAS verification failed: C[0] = {} expected {}",
                            C[0], expected);
                return EXIT_FAILURE;
        }

        const auto metrics = calculate_metrics(matrix_size);
        std::cout << core::write(matrix_report{
                         .mode = "openblas",
                         .matrix_size = matrix_size,
                         .seconds = best_time,
                         .gflops = metrics.flops / best_time / 1e9,
                         .bandwidth_gbs =
                             metrics.total_data_size_gb / best_time,
                     })
                  << '\n';

        return EXIT_SUCCESS;
}
