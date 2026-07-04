#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "header.hh"

namespace config {
constexpr std::size_t default_size = 4096;
constexpr std::size_t default_repeat = 100;
constexpr std::uint64_t default_seed = 7;
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

struct csr_matrix {
    std::size_t rows{};
    std::vector<std::size_t> row_offsets;
    std::vector<std::uint32_t> column_indices;
    std::vector<float> values;
};

void setup_threads();
csr_matrix make_poisson5_matrix(std::size_t grid_size);
csr_matrix make_powerlaw_matrix(std::size_t rows, std::uint64_t seed);
std::vector<std::size_t> make_balanced_row_splits(const csr_matrix& matrix, int threads);
void spmv_csr_balanced(const csr_matrix& matrix, const std::vector<float>& x,
                       std::vector<float>& y, const std::vector<std::size_t>& row_splits);
void spmv_csr_reference(const csr_matrix& matrix, const std::vector<float>& x,
                        std::vector<float>& y);

int main(int argc, char* argv[]) {
    CLI::App app{"CSR SpMV balanced"};
    std::size_t size = config::default_size;
    std::size_t repeat = config::default_repeat;
    std::uint64_t seed = config::default_seed;
    std::string matrix_name = "powerlaw";
    bool check = false;

    app.add_option("-n,--size", size, "Matrix size, or grid width for poisson5")
        ->check(CLI::PositiveNumber);
    app.add_option("-r,--repeat", repeat, "SpMV repeat count")->check(CLI::PositiveNumber);
    app.add_option("--matrix", matrix_name, "Synthetic matrix: poisson5 or powerlaw");
    app.add_option("--seed", seed, "Random seed for irregular matrix generation");
    app.add_flag("--check", check, "Compare against a serial reference result");
    CLI11_PARSE(app, argc, argv);

    setup_threads();

    csr_matrix matrix;
    if (matrix_name == "poisson5") {
        matrix = make_poisson5_matrix(size);
    } else if (matrix_name == "powerlaw") {
        matrix = make_powerlaw_matrix(size, seed);
    } else {
        spdlog::error("unknown matrix type: {}", matrix_name);
        return EXIT_FAILURE;
    }

    const int threads = omp_get_max_threads();
    const auto row_splits = make_balanced_row_splits(matrix, threads);
    std::vector<float> x(matrix.rows, 1.0F);
    std::vector<float> y(matrix.rows);

    timer.start();
    for (std::size_t iteration = 0; iteration < repeat; iteration++) {
        spmv_csr_balanced(matrix, x, y, row_splits);
    }
    timer.end();

    const double checksum = std::reduce(y.begin(), y.end(), 0.0);
    if (check) {
        std::vector<float> reference(matrix.rows);
        spmv_csr_reference(matrix, x, reference);

        double max_error = 0.0;
        for (std::size_t row = 0; row < matrix.rows; row++) {
            max_error =
                std::max(max_error, std::abs(static_cast<double>(y[row] - reference[row])));
        }

        if (max_error > 1.0e-5) {
            spdlog::error("check failed: max error = {}", max_error);
            return EXIT_FAILURE;
        }
        spdlog::info("check passed, max error = {}", max_error);
    }

    constexpr double giga = 1e9;
    const double seconds = timer.elapsed();
    const double nonzeros = static_cast<double>(matrix.values.size());
    const double total_nonzeros = nonzeros * static_cast<double>(repeat);
    const double bytes = total_nonzeros * (sizeof(float) + sizeof(std::uint32_t)) +
                         total_nonzeros * sizeof(float) +
                         static_cast<double>(matrix.rows * repeat) * sizeof(float);

    std::cout << core::write(threaded_spmv_csr_report{
                     .mode = "balanced",
                     .matrix = std::string_view{matrix_name},
                     .rows = matrix.rows,
                     .nonzeros = matrix.values.size(),
                     .nonzeros_per_row = nonzeros / matrix.rows,
                     .threads = threads,
                     .repeat = repeat,
                     .seconds = seconds,
                     .throughput_gnnzs = total_nonzeros / seconds / giga,
                     .bandwidth_gbs = bytes / seconds / giga,
                     .checksum = checksum,
                 })
              << '\n';

    return EXIT_SUCCESS;
}

csr_matrix make_poisson5_matrix(std::size_t grid_size) {
    csr_matrix matrix;
    matrix.rows = grid_size * grid_size;
    matrix.row_offsets.resize(matrix.rows + 1);
    matrix.column_indices.reserve(matrix.rows * 5);
    matrix.values.reserve(matrix.rows * 5);

    const auto append = [&matrix](std::size_t column, float value) {
        matrix.column_indices.push_back(static_cast<std::uint32_t>(column));
        matrix.values.push_back(value);
    };

    for (std::size_t i = 0; i < grid_size; i++) {
        for (std::size_t j = 0; j < grid_size; j++) {
            const std::size_t row = (i * grid_size) + j;
            matrix.row_offsets[row] = matrix.values.size();

            append(row, 4.0F);
            if (i > 0) {
                append(row - grid_size, -1.0F);
            }
            if (i + 1 < grid_size) {
                append(row + grid_size, -1.0F);
            }
            if (j > 0) {
                append(row - 1, -1.0F);
            }
            if (j + 1 < grid_size) {
                append(row + 1, -1.0F);
            }
        }
    }

    matrix.row_offsets[matrix.rows] = matrix.values.size();
    return matrix;
}

csr_matrix make_powerlaw_matrix(std::size_t rows, std::uint64_t seed) {
    csr_matrix matrix;
    matrix.rows = rows;
    matrix.row_offsets.resize(rows + 1);
    if (rows == 0) {
        return matrix;
    }

    const auto row_length_for = [](std::size_t row) {
        if (row % 1000 == 0) {
            return std::size_t{4096};
        }
        if (row % 100 == 0) {
            return std::size_t{256};
        }
        return std::size_t{4};
    };

    std::size_t nonzeros = 0;
    for (std::size_t row = 0; row < rows; row++) {
        nonzeros += row_length_for(row);
    }
    matrix.column_indices.reserve(nonzeros);
    matrix.values.reserve(nonzeros);

    std::mt19937_64 random_number_generator(seed);
    std::uniform_int_distribution<std::size_t> column_distribution(0, rows - 1);

    for (std::size_t row = 0; row < rows; row++) {
        matrix.row_offsets[row] = matrix.values.size();
        const std::size_t row_length = row_length_for(row);
        const float value = 1.0F / static_cast<float>(row_length);

        for (std::size_t index = 0; index < row_length; index++) {
            matrix.column_indices.push_back(
                static_cast<std::uint32_t>(column_distribution(random_number_generator)));
            matrix.values.push_back(value);
        }
    }

    matrix.row_offsets[rows] = matrix.values.size();
    return matrix;
}

std::vector<std::size_t> make_balanced_row_splits(const csr_matrix& matrix, int threads) {
    const auto thread_count = static_cast<std::size_t>(std::max(threads, 1));
    std::vector<std::size_t> row_splits(thread_count + 1);
    row_splits.front() = 0;
    row_splits.back() = matrix.rows;

    const std::size_t total_nonzeros = matrix.row_offsets.back();
    for (std::size_t thread = 1; thread < thread_count; thread++) {
        const std::size_t target = (total_nonzeros * thread) / thread_count;
        const auto split = static_cast<std::size_t>(std::distance(
            matrix.row_offsets.begin(),
            std::lower_bound(matrix.row_offsets.begin(), matrix.row_offsets.end(), target)));
        row_splits[thread] = std::clamp(split, row_splits[thread - 1], matrix.rows);
    }

    return row_splits;
}

void spmv_csr_balanced(const csr_matrix& matrix, const std::vector<float>& x,
                       std::vector<float>& y, const std::vector<std::size_t>& row_splits) {
#pragma omp parallel for schedule(static)
    for (std::size_t split = 0; split < row_splits.size() - 1; split++) {
        for (std::size_t row = row_splits[split]; row < row_splits[split + 1]; row++) {
            float accumulator = 0.0F;
            const std::size_t begin = matrix.row_offsets[row];
            const std::size_t end = matrix.row_offsets[row + 1];

            for (std::size_t offset = begin; offset < end; offset++) {
                const std::uint32_t column = matrix.column_indices[offset];
                accumulator += matrix.values[offset] * x[column];
            }

            y[row] = accumulator;
        }
    }
}

void spmv_csr_reference(const csr_matrix& matrix, const std::vector<float>& x,
                        std::vector<float>& y) {
    for (std::size_t row = 0; row < matrix.rows; row++) {
        float accumulator = 0.0F;
        const std::size_t begin = matrix.row_offsets[row];
        const std::size_t end = matrix.row_offsets[row + 1];

        for (std::size_t offset = begin; offset < end; offset++) {
            const std::uint32_t column = matrix.column_indices[offset];
            accumulator += matrix.values[offset] * x[column];
        }

        y[row] = accumulator;
    }
}

void setup_threads() {
    const auto* const omp_num_threads = std::getenv("OMP_NUM_THREADS");
    auto threads_count = static_cast<bool>(omp_num_threads)
                             ? std::atoi(omp_num_threads)
                             : static_cast<int>(std::thread::hardware_concurrency());
    threads_count = std::max(threads_count, 1);
    omp_set_dynamic(0);
    omp_set_num_threads(threads_count);
    spdlog::info("threads = {}", threads_count);
}
