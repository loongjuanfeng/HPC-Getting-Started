#include <CLI/CLI.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <limits>
#include <random>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hh"

void check_cuda(cudaError_t result, const char* call) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(call) + " failed: " + cudaGetErrorString(result));
    }
}

#define CHECK_CUDA(call) check_cuda((call), #call)

template <typename Type, std::size_t vector_count> auto create_vectors(std::size_t vector_size) {
    std::array<std::vector<Type>, vector_count> vectors;
    for (auto& vector : vectors) {
        vector.resize(vector_size);
    }
    return vectors;
}

class CudaEventTimer {
    cudaEvent_t time_start{};
    cudaEvent_t time_end{};

public:
    CudaEventTimer() {
        CHECK_CUDA(cudaEventCreate(&time_start));
        CHECK_CUDA(cudaEventCreate(&time_end));
    }

    ~CudaEventTimer() {
        cudaEventDestroy(time_end);
        cudaEventDestroy(time_start);
    }

    void start() { CHECK_CUDA(cudaEventRecord(time_start)); }

    void end() {
        CHECK_CUDA(cudaEventRecord(time_end));
        CHECK_CUDA(cudaEventSynchronize(time_end));
    }

    [[nodiscard]] double elapsed() const {
        float elapsed_ms{};
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, time_start, time_end));
        return static_cast<double>(elapsed_ms) / 1000.0;
    }
};

__global__ void vector_addition_kernel(const float* vector_1, const float* vector_2,
                                       float* sum_vector, std::size_t vector_size) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < vector_size) {
        sum_vector[index] = vector_1[index] + vector_2[index];
    }
}

int main(int argc, char* argv[]) try {
    CLI::App app{"CUDA vector addition - HPC starter"};
    std::size_t vector_size{config::default_vector_size};
    int block_size{config::default_block_size};
    app.add_option("-s,--size", vector_size, "Number of elements")->check(CLI::PositiveNumber);
    app.add_option("-b,--block-size", block_size, "CUDA threads per block")
        ->check(CLI::Range(1, 1024));
    CLI11_PARSE(app, argc, argv);

    auto [vector_1, vector_2, sum_vector] = create_vectors<float, 3>(vector_size);

    std::mt19937 random_number_generator(std::random_device{}());
    std::normal_distribution<float> normal_distribution{1.0};
    for (std::size_t i = 0; i < vector_size; i++) {
        vector_1[i] = normal_distribution(random_number_generator);
        vector_2[i] = normal_distribution(random_number_generator);
    }

    float* device_vector_1{};
    float* device_vector_2{};
    float* device_sum_vector{};
    const auto bytes = vector_size * sizeof(float);
    CHECK_CUDA(cudaMalloc(&device_vector_1, bytes));
    CHECK_CUDA(cudaMalloc(&device_vector_2, bytes));
    CHECK_CUDA(cudaMalloc(&device_sum_vector, bytes));

    CHECK_CUDA(cudaMemcpy(device_vector_1, vector_1.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_vector_2, vector_2.data(), bytes, cudaMemcpyHostToDevice));

    CudaEventTimer timer;
    const auto grid_size =
        static_cast<int>((vector_size + static_cast<std::size_t>(block_size) - 1) / block_size);

    timer.start();
    vector_addition_kernel<<<grid_size, block_size>>>(device_vector_1, device_vector_2,
                                                      device_sum_vector, vector_size);
    CHECK_CUDA(cudaGetLastError());
    timer.end();

    CHECK_CUDA(cudaMemcpy(sum_vector.data(), device_sum_vector, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_sum_vector));
    CHECK_CUDA(cudaFree(device_vector_2));
    CHECK_CUDA(cudaFree(device_vector_1));

    double vector_sum{0.0};
    for (const auto value : sum_vector) {
        vector_sum += value;
    }

    const auto expected = vector_1.front() + vector_2.front();
    if (std::abs(sum_vector.front() - expected) > std::numeric_limits<float>::epsilon() * 16.0F) {
        spdlog::error("CUDA verification failed: sum_vector[0] = {} expected {}",
                      sum_vector.front(), expected);
        return EXIT_FAILURE;
    }

    auto report_logger = spdlog::stdout_color_mt("REPORT");

    const double vector_total_size = 3.0 * static_cast<double>(vector_size * sizeof(float)) / 1e9;
    report_logger->info("=== CUDA Vector Addition ===");
    report_logger->info("{:>12} = {:>10.0f}", "size", static_cast<double>(vector_size));
    report_logger->info("{:>12} = {:>10.0f}", "block size", static_cast<double>(block_size));
    report_logger->info("{:>12} = {:>10.4f} s", "time used", timer.elapsed());
    report_logger->info("{:>12} = {:>10.4f} GB/s", "bandwidth",
                        vector_total_size / timer.elapsed());
    report_logger->info("{:>12} = {:>10.4f}", "mean", vector_sum / vector_size);

    return EXIT_SUCCESS;
} catch (const std::exception& exception) {
    spdlog::error("{}", exception.what());
    return EXIT_FAILURE;
}
