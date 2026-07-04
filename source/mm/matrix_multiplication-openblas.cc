#include "cblas.h"
#include "spdlog/common.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <random>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <CLI/CLI.hpp>

#include "header.hh"

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
                    matrix_size, 1.0F, A.data(), matrix_size, B.data(), matrix_size, 1.0F, C.data(),
                    matrix_size);

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
    const double flops = 2.0 * static_cast<double>(matrix_size) * static_cast<double>(matrix_size) *
                         static_cast<double>(matrix_size);
    report_logger->info("=== OpenBLAS sgemm ===");
    report_logger->info("{:>12} = {:>10.0f} x {:<10.0f}", "size", static_cast<double>(matrix_size),
                        static_cast<double>(matrix_size));
    report_logger->info("{:>12} = {:>10.4f} s", "time used", best_time);
    report_logger->info("{:>12} = {:>10.4f} GFLOPS", "compute", flops / best_time / giga_bytes);
    report_logger->info("{:>12} = {:>10.4f} GB/s", "bandwidth", total_data_size / best_time);

    return EXIT_SUCCESS;
}
