#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include "header.hh"
#include "log.hh"

namespace {
void check_cuda(cudaError_t result, const char* call) {
        if (result != cudaSuccess) {
                throw std::runtime_error(std::string(call) + " failed: " +
                                         cudaGetErrorString(result));
        }
}

void check_cublas(cublasStatus_t result, const char* call) {
        if (result != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error(
                    std::string(call) + " failed with cuBLAS status " +
                    std::to_string(static_cast<int>(result)));
        }
}

#define CHECK_CUDA(call) check_cuda((call), #call)
#define CHECK_CUBLAS(call) check_cublas((call), #call)

class cuda_buffer {
public:
        explicit cuda_buffer(const std::size_t bytes) : bytes_{bytes} {
                CHECK_CUDA(cudaMalloc(&pointer_, bytes_));
        }

        cuda_buffer(const cuda_buffer&) = delete;
        cuda_buffer& operator=(const cuda_buffer&) = delete;

        cuda_buffer(cuda_buffer&& other) noexcept
            : pointer_{other.pointer_}, bytes_{other.bytes_} {
                other.pointer_ = nullptr;
                other.bytes_ = 0;
        }

        cuda_buffer& operator=(cuda_buffer&& other) noexcept {
                if (this != &other) {
                        release();
                        pointer_ = other.pointer_;
                        bytes_ = other.bytes_;
                        other.pointer_ = nullptr;
                        other.bytes_ = 0;
                }
                return *this;
        }

        ~cuda_buffer() { release(); }

        [[nodiscard]] float* get() noexcept {
                return static_cast<float*>(pointer_);
        }

        [[nodiscard]] const float* get() const noexcept {
                return static_cast<const float*>(pointer_);
        }

        [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
        void release() noexcept {
                if (pointer_ != nullptr) {
                        cudaFree(pointer_);
                }
        }

        void* pointer_{};
        std::size_t bytes_{};
};

class cublas_handle {
public:
        cublas_handle() { CHECK_CUBLAS(cublasCreate(&handle_)); }

        cublas_handle(const cublas_handle&) = delete;
        cublas_handle& operator=(const cublas_handle&) = delete;

        ~cublas_handle() {
                if (handle_ != nullptr) {
                        cublasDestroy(handle_);
                }
        }

        [[nodiscard]] cublasHandle_t get() const noexcept { return handle_; }

private:
        cublasHandle_t handle_{};
};

class cuda_event_timer {
public:
        cuda_event_timer() {
                CHECK_CUDA(cudaEventCreate(&start_));
                CHECK_CUDA(cudaEventCreate(&stop_));
        }

        cuda_event_timer(const cuda_event_timer&) = delete;
        cuda_event_timer& operator=(const cuda_event_timer&) = delete;

        ~cuda_event_timer() {
                cudaEventDestroy(stop_);
                cudaEventDestroy(start_);
        }

        void start() { CHECK_CUDA(cudaEventRecord(start_)); }

        void end() {
                CHECK_CUDA(cudaEventRecord(stop_));
                CHECK_CUDA(cudaEventSynchronize(stop_));
        }

        [[nodiscard]] double elapsed() const {
                float milliseconds{};
                CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start_, stop_));
                return static_cast<double>(milliseconds) / 1000.0;
        }

private:
        cudaEvent_t start_{};
        cudaEvent_t stop_{};
};

void initialize_inputs(matrix_vector<float>& A, matrix_vector<float>& B,
                       const std::size_t matrix_size) {
        std::mt19937 random_number_generator(std::random_device{}());
        std::normal_distribution<float> normal_distribution{1.0};
        for (std::size_t i = 0; i < matrix_size * matrix_size; i++) {
                A[i] = normal_distribution(random_number_generator);
                B[i] = normal_distribution(random_number_generator);
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

        core::ERROR("cuBLAS verification failed: C[0] = {} expected {}", C[0],
                    expected);
        return false;
}
}  // namespace

int main(int argc, char* argv[]) try {
        CLI::App app{"cuBLAS matrix multiplication - HPC"};
        std::size_t matrix_size{config::default_matrix_size};
        app.add_option("-s,--size", matrix_size, "Matrix dimension (N x N)");
        CLI11_PARSE(app, argc, argv);

        auto [A, B, C] = create_matrix<float, 3>(matrix_size);
        initialize_inputs(A, B, matrix_size);

        const auto bytes = matrix_size * matrix_size * sizeof(float);
        cuda_buffer device_A{bytes};
        cuda_buffer device_B{bytes};
        cuda_buffer device_C{bytes};

        CHECK_CUDA(cudaMemcpy(device_A.get(), A.data(), bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(device_B.get(), B.data(), bytes,
                              cudaMemcpyHostToDevice));

        cublas_handle handle;
        cuda_event_timer timer;
        double best_time = std::numeric_limits<double>::max();

        const auto n = static_cast<int>(matrix_size);
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;

        for (std::size_t loop_index = 0; loop_index < config::calculation_loop;
             loop_index++) {
                CHECK_CUDA(cudaMemset(device_C.get(), 0, bytes));

                timer.start();
                CHECK_CUBLAS(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                                         n, n, n, &alpha, device_B.get(), n,
                                         device_A.get(), n, &beta,
                                         device_C.get(), n));
                timer.end();

                best_time = std::min(best_time, timer.elapsed());
                core::INFO("loop {}, time: {}", loop_index, timer.elapsed());
        }

        CHECK_CUDA(cudaMemcpy(C.data(), device_C.get(), bytes,
                              cudaMemcpyDeviceToHost));

        if (!verify_first_element(A, B, C, matrix_size)) {
                return EXIT_FAILURE;
        }

        const auto metrics = calculate_metrics(matrix_size);
        std::cout << core::write(matrix_report{
                         .mode = "cublas",
                         .matrix_size = matrix_size,
                         .seconds = best_time,
                         .gflops = metrics.flops / best_time / 1e9,
                         .bandwidth_gbs =
                             metrics.total_data_size_gb / best_time,
                     })
                  << '\n';

        return EXIT_SUCCESS;
} catch (const std::exception& exception) {
        core::ERROR("{}", exception.what());
        return EXIT_FAILURE;
}
