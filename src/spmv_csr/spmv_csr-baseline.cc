#include <algorithm>
#include <chrono>
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
    std::vector<std::size_t> column_indices;
    std::vector<float> values;
};

csr_matrix make_poisson5_matrix(std::size_t grid_size);
csr_matrix make_powerlaw_matrix(std::size_t rows, std::uint64_t seed);
void spmv_csr_baseline(const csr_matrix& matrix, const std::vector<float>& x,
                       std::vector<float>& y);

int main(int argc, char* argv[]) {
    CLI::App app{"CSR SpMV baseline template"};
    std::size_t size = config::default_size;
    std::size_t repeat = config::default_repeat;
    std::uint64_t seed = config::default_seed;
    std::string matrix_name = "poisson5";
    bool check = false;

    app.add_option("-n,--size", size, "Matrix size, or grid width for poisson5");
    app.add_option("-r,--repeat", repeat, "SpMV repeat count");
    app.add_option("--matrix", matrix_name, "Synthetic matrix: {poisson5,powerlaw}");
    app.add_option("--seed", seed, "Random seed for irregular matrix generation");
    app.add_flag("--check", check, "Check result after the baseline kernel is implemented");
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
        spdlog::warn("check is wired, but the baseline kernel is still a TODO template");
    }

    const auto report_logger = spdlog::stdout_color_mt("REPORT");
    constexpr double giga = 1e9;
    const double seconds = timer.elapsed();
    const auto nonzeros = static_cast<double>(matrix.values.size());
    const double total_nonzeros = nonzeros * static_cast<double>(repeat);
    const double bytes = (total_nonzeros * (sizeof(float) + sizeof(std::uint32_t))) +
                         (total_nonzeros * sizeof(float)) +
                         (static_cast<double>(matrix.rows * repeat) * sizeof(float));

    report_logger->info("=== CSR SpMV Baseline Template ===");
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

    // Instructions:
    // 1. In this generator, 'grid_size' is the width of a square 2D grid.
    //    The sparse matrix has grid_size * grid_size rows.
    // 2. Map a grid point (i, j) to a matrix row with:
    //        row = i * grid_size + j
    // 3. Visit rows in increasing row order. At the start of each row, set:
    //        matrix.row_offsets[row] = matrix.values.size()
    // 4. Append the center entry first:
    //        column = row
    //        value  = 4.0F
    // 5. Append neighbor entries only when the neighbor exists:
    //        up    exists when i > 0
    //        down  exists when i + 1 < grid_size
    //        left  exists when j > 0
    //        right exists when j + 1 < grid_size
    //    Use value = -1.0F for each neighbor.
    // 6. For every appended nonzero, push one column id and one value.
    //    column_indices.size() must always equal values.size().
    // 7. After all rows are done, set:
    //        matrix.row_offsets[matrix.rows] = matrix.values.size()
    // 8. Sanity checks to print or assert while debugging:
    //        row_offsets.size() == rows + 1
    //        row_offsets.front() == 0
    //        row_offsets.back() == values.size()
    //        column_indices.size() == values.size()
    matrix.row_offsets.back() = 0;
    return matrix;
}

csr_matrix make_powerlaw_matrix(std::size_t rows, std::uint64_t seed) {
    csr_matrix matrix;
    matrix.rows = rows;
    matrix.row_offsets.resize(rows + 1);

    // Instructions:
    // 1. Build this after poisson5 and the baseline SpMV kernel are correct.
    // 2. The goal is not a realistic matrix yet. The goal is uneven row work:
    //        most rows have a few nonzeros
    //        a small number of rows have many nonzeros
    // 3. Use std::mt19937_64 seeded by 'seed' so benchmark results are repeatable.
    // 4. For each row:
    //        set row_offsets[row] = values.size()
    //        choose a row length
    //        append that many random columns in [0, rows)
    //        append one float value for each column
    // 5. A simple first row-length rule:
    //        if row % 1000 == 0, use 4096 nonzeros
    //        else if row % 100 == 0, use 256 nonzeros
    //        else use 4 nonzeros
    // 6. Keep column ids valid. Do not worry about duplicate columns at first.
    //    Duplicate columns are legal in CSR, but they add together mathematically.
    // 7. Use small values such as 1.0F / row_length to keep checksums readable.
    // 8. Finish by setting row_offsets[rows] = values.size().
    // 9. This matrix should make naive OpenMP static row scheduling look worse.
    (void)seed;
    matrix.row_offsets.back() = 0;
    return matrix;
}

void spmv_csr_baseline(const csr_matrix& matrix, const std::vector<float>& x,
                       std::vector<float>& y) {
    // Instructions:
    // 1. This is y = A * x where A is stored in CSR format.
    // 2. Loop over rows serially:
    //        for row in [0, matrix.rows)
    // 3. For each row, get its nonzero range:
    //        begin = row_offsets[row]
    //        end   = row_offsets[row + 1]
    // 4. Start a local float accumulator at 0.0F.
    // 5. Loop over k in [begin, end):
    //        column = column_indices[k]
    //        value  = values[k]
    //        accumulator += value * x[column]
    // 6. After the inner loop, write:
    //        y[row] = accumulator
    // 7. Do not write to y[row] before the accumulator is complete.
    // 8. Do not use OpenMP in this baseline file.
    // 9. Useful first test:
    //        poisson5 with x filled with 1.0F
    //        interior rows should produce 0.0F because 4 - 1 - 1 - 1 - 1 = 0
    //        edge rows produce positive values because they have fewer neighbors
    // 10. After this works, use this function as the reference for OpenMP checks.
    std::fill(y.begin(), y.end(), 0.0F);
    (void)matrix;
    (void)x;
}
