#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

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

csr_matrix make_poisson5_matrix(std::size_t grid_size);
csr_matrix make_powerlaw_matrix(std::size_t rows, std::uint64_t seed);
void spmv_csr_baseline(const csr_matrix& matrix, const std::vector<float>& x,
                       std::vector<float>& y);

int main(int argc, char* argv[]) {
    CLI::App app{"CSR SpMV baseline"};
    std::size_t size = config::default_size;
    std::size_t repeat = config::default_repeat;
    std::uint64_t seed = config::default_seed;
    std::string matrix_name = "poisson5";
    bool check = false;

    app.add_option("-n,--size", size, "Matrix size, or grid width for poisson5")
        ->check(CLI::PositiveNumber);
    app.add_option("-r,--repeat", repeat, "SpMV repeat count")->check(CLI::PositiveNumber);
    app.add_option("--matrix", matrix_name, "Synthetic matrix: {poisson5,powerlaw}");
    app.add_option("--seed", seed, "Random seed for irregular matrix generation");
    app.add_flag("--check", check, "Validate the computed checksum");
    CLI11_PARSE(app, argc, argv);

    csr_matrix matrix;
    if (matrix_name == "poisson5") {
        matrix = make_poisson5_matrix(size);
    } else if (matrix_name == "powerlaw") {
        matrix = make_powerlaw_matrix(size, seed);
    } else {
        spdlog::error("unknown matrix type: {}", matrix_name);
        return EXIT_FAILURE;
    }

    std::vector<float> x(matrix.rows, 1.0F);
    std::vector<float> y(matrix.rows);

    timer.start();
    for (std::size_t iteration = 0; iteration < repeat; iteration++) {
        spmv_csr_baseline(matrix, x, y);
    }
    timer.end();

    const double checksum = std::reduce(y.begin(), y.end(), 0.0);
    if (check) {
        const double expected_checksum =
            matrix_name == "poisson5" ? static_cast<double>(4 * size)
                                      : static_cast<double>(matrix.rows);
        const double tolerance = 1.0e-4 * std::max(1.0, expected_checksum);
        if (std::abs(checksum - expected_checksum) > tolerance) {
            spdlog::error("check failed: checksum = {}, expected {}", checksum,
                          expected_checksum);
            return EXIT_FAILURE;
        }
        spdlog::info("check passed");
    }

    const auto report_logger = spdlog::stdout_color_mt("REPORT");
    constexpr double giga = 1e9;
    const double seconds = timer.elapsed();
    const auto nonzeros = static_cast<double>(matrix.values.size());
    const double total_nonzeros = nonzeros * static_cast<double>(repeat);
    const double bytes = (total_nonzeros * (sizeof(float) + sizeof(std::uint32_t))) +
                         (total_nonzeros * sizeof(float)) +
                         (static_cast<double>(matrix.rows * repeat) * sizeof(float));

    report_logger->info("=== CSR SpMV Baseline ===");
    report_logger->info("{:>12} = {:>10}", "matrix", matrix_name);
    report_logger->info("{:>12} = {:>10}", "rows", matrix.rows);
    report_logger->info("{:>12} = {:>10}", "nonzeros", matrix.values.size());
    report_logger->info("{:>12} = {:>10.2f}", "nnz/row", nonzeros / matrix.rows);
    report_logger->info("{:>12} = {:>10}", "repeat", repeat);
    report_logger->info("{:>12} = {:>10.4f} s", "time", seconds);
    report_logger->info("{:>12} = {:>10.4f} GNNZ/s", "throughput", total_nonzeros / seconds / giga);
    report_logger->info("{:>12} = {:>10.4f} GB/s", "bandwidth", bytes / seconds / giga);
    report_logger->info("{:>12} = {:>10.4f}", "checksum", checksum);

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

void spmv_csr_baseline(const csr_matrix& matrix, const std::vector<float>& x,
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
