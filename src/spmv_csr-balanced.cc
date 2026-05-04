#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <omp.h>
#include <random>
#include <string>
#include <thread>
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

void setup_threads();
csr_matrix make_poisson5_matrix(std::size_t grid_size);
csr_matrix make_powerlaw_matrix(std::size_t rows, std::uint64_t seed);
std::vector<std::size_t> make_balanced_row_splits(const csr_matrix& matrix, int threads);
void spmv_csr_balanced(const csr_matrix& matrix, const std::vector<float>& x,
                       std::vector<float>& y, const std::vector<std::size_t>& row_splits);

int main(int argc, char* argv[]) {
    CLI::App app{"CSR SpMV balanced template"};
    std::size_t size = config::default_size;
    std::size_t repeat = config::default_repeat;
    std::uint64_t seed = config::default_seed;
    std::string matrix_name = "powerlaw";
    bool check = false;

    app.add_option("-n,--size", size, "Matrix size, or grid width for poisson5");
    app.add_option("-r,--repeat", repeat, "SpMV repeat count");
    app.add_option("--matrix", matrix_name, "Synthetic matrix: poisson5 or powerlaw");
    app.add_option("--seed", seed, "Random seed for irregular matrix generation");
    app.add_flag("--check", check, "Check result after the baseline kernel is implemented");
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
        spdlog::warn("check is wired, but the serial reference kernel is still a TODO template");
    }

    const auto report_logger = spdlog::stdout_color_mt("REPORT");
    constexpr double giga = 1e9;
    const double seconds = timer.elapsed();
    const double nonzeros = static_cast<double>(matrix.values.size());
    const double total_nonzeros = nonzeros * static_cast<double>(repeat);
    const double bytes = total_nonzeros * (sizeof(float) + sizeof(std::uint32_t)) +
                         total_nonzeros * sizeof(float) +
                         static_cast<double>(matrix.rows * repeat) * sizeof(float);

    report_logger->info("=== CSR SpMV Balanced Template ===");
    report_logger->info("{:>12} = {:>10}", "matrix", matrix_name);
    report_logger->info("{:>12} = {:>10}", "rows", matrix.rows);
    report_logger->info("{:>12} = {:>10}", "nonzeros", matrix.values.size());
    report_logger->info("{:>12} = {:>10.2f}", "nnz/row", nonzeros / matrix.rows);
    report_logger->info("{:>12} = {:>10}", "threads", threads);
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
    // 1. Copy your correct baseline matrix generator here.
    // 2. Balanced scheduling should produce the same checksum as baseline.
    matrix.row_offsets.back() = 0;
    return matrix;
}

csr_matrix make_powerlaw_matrix(std::size_t rows, std::uint64_t seed) {
    csr_matrix matrix;
    matrix.rows = rows;
    matrix.row_offsets.resize(rows + 1);

    // Instructions:
    // 1. Copy your correct baseline irregular generator here.
    // 2. Make row lengths uneven enough that static row scheduling is bad.
    // 3. This file should improve the powerlaw case more than poisson5.
    (void)seed;
    matrix.row_offsets.back() = 0;
    return matrix;
}

std::vector<std::size_t> make_balanced_row_splits(const csr_matrix& matrix, int threads) {
    std::vector<std::size_t> row_splits(static_cast<std::size_t>(threads) + 1);
    row_splits.front() = 0;
    row_splits.back() = matrix.rows;

    // Instructions:
    // 1. Use row_offsets to estimate work per row from nonzero counts.
    // 2. For each thread id, find the row where cumulative nnz reaches its target.
    // 3. row_splits[t] and row_splits[t + 1] define rows owned by thread t.
    // 4. Keep splits monotonic and always include 0 and matrix.rows.
    return row_splits;
}

void spmv_csr_balanced(const csr_matrix& matrix, const std::vector<float>& x,
                       std::vector<float>& y, const std::vector<std::size_t>& row_splits) {
    // Instructions:
    // 1. Start one OpenMP parallel region.
    // 2. Each thread reads its row range from row_splits.
    // 3. Compute normal CSR SpMV for only those rows.
    // 4. Each thread writes unique y[row] entries, so no atomics are needed.
    // 5. Compare this against spmv_csr-openmp on the powerlaw matrix.
#pragma omp parallel for schedule(static)
    for (std::size_t row = 0; row < y.size(); row++) {
        y[row] = 0.0F;
    }
    (void)matrix;
    (void)x;
    (void)row_splits;
}

void setup_threads() {
    const auto* const omp_num_threads = std::getenv("OMP_NUM_THREADS");
    const auto threads_count = static_cast<bool>(omp_num_threads)
                                   ? std::atoi(omp_num_threads)
                                   : static_cast<int>(std::thread::hardware_concurrency());
    omp_set_num_threads(threads_count);
    spdlog::info("threads = {}", threads_count);
}
