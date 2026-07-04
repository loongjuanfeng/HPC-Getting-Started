#include "cuda_glibc_compat.hh"

#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#include "header.hh"

namespace {
void check_cuda(cudaError_t result, const char* call) {
        if (result != cudaSuccess) {
                throw std::runtime_error(std::string(call) + " failed: " +
                                         cudaGetErrorString(result));
        }
}

#define CHECK_CUDA(call) check_cuda((call), #call)

struct options {
        std::size_t vector_size{config::default_vector_size};
        int block_size{config::default_block_size};
};

void print_usage(std::string_view program) {
        core::INFO("Usage: {} [--size N] [--block-size N]", program);
}

template <typename Type>
Type parse_positive_integer(std::string_view value, std::string_view name) {
        Type parsed{};
        const auto* begin = value.data();
        const auto* end = value.data() + value.size();
        const auto [position, error] = std::from_chars(begin, end, parsed);
        if (error != std::errc{} || position != end || parsed <= 0) {
                throw std::runtime_error("invalid value for " +
                                         std::string(name) + ": " +
                                         std::string(value));
        }
        return parsed;
}

options parse_options(int argc, char* argv[]) {
        options parsed_options;

        const auto read_value = [&](int& index,
                                    std::string_view name) -> std::string_view {
                if (index + 1 >= argc) {
                        throw std::runtime_error("missing value for " +
                                                 std::string(name));
                }
                ++index;
                return argv[index];
        };

        for (int index = 1; index < argc; ++index) {
                const std::string_view argument{argv[index]};
                if (argument == "-h" || argument == "--help") {
                        print_usage(argv[0]);
                        std::exit(EXIT_SUCCESS);
                }
                if (argument == "-s" || argument == "--size") {
                        parsed_options.vector_size =
                            parse_positive_integer<std::size_t>(
                                read_value(index, argument), argument);
                        continue;
                }
                if (argument.starts_with("--size=")) {
                        parsed_options.vector_size =
                            parse_positive_integer<std::size_t>(
                                argument.substr(7), "--size");
                        continue;
                }
                if (argument == "-b" || argument == "--block-size") {
                        parsed_options.block_size = parse_positive_integer<int>(
                            read_value(index, argument), argument);
                        continue;
                }
                if (argument.starts_with("--block-size=")) {
                        parsed_options.block_size = parse_positive_integer<int>(
                            argument.substr(13), "--block-size");
                        continue;
                }
                throw std::runtime_error("unknown option: " +
                                         std::string(argument));
        }

        if (parsed_options.block_size > 1024) {
                throw std::runtime_error(
                    "invalid value for --block-size: expected 1 through 1024");
        }

        return parsed_options;
}

class cuda_event_timer {
public:
        cuda_event_timer() {
                CHECK_CUDA(cudaEventCreate(&time_start));
                CHECK_CUDA(cudaEventCreate(&time_end));
        }

        cuda_event_timer(const cuda_event_timer&) = delete;
        cuda_event_timer& operator=(const cuda_event_timer&) = delete;

        ~cuda_event_timer() {
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
                CHECK_CUDA(
                    cudaEventElapsedTime(&elapsed_ms, time_start, time_end));
                return static_cast<double>(elapsed_ms) / 1000.0;
        }

private:
        cudaEvent_t time_start{};
        cudaEvent_t time_end{};
};

class cuda_buffer {
public:
        explicit cuda_buffer(const std::size_t bytes) {
                CHECK_CUDA(cudaMalloc(&pointer_, bytes));
        }

        cuda_buffer(const cuda_buffer&) = delete;
        cuda_buffer& operator=(const cuda_buffer&) = delete;

        ~cuda_buffer() {
                if (pointer_ != nullptr) {
                        cudaFree(pointer_);
                }
        }

        [[nodiscard]] float* get() noexcept {
                return static_cast<float*>(pointer_);
        }

private:
        void* pointer_{};
};

void initialize_inputs(vector_storage<float>& vector_1,
                       vector_storage<float>& vector_2,
                       const std::size_t vector_size) {
        std::mt19937 random_number_generator(std::random_device{}());
        std::normal_distribution<float> normal_distribution{1.0};
        for (std::size_t i = 0; i < vector_size; i++) {
                vector_1[i] = normal_distribution(random_number_generator);
                vector_2[i] = normal_distribution(random_number_generator);
        }
}

__global__ void vector_addition_kernel(const float* vector_1,
                                       const float* vector_2, float* sum_vector,
                                       std::size_t vector_size) {
        const auto index =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index < vector_size) {
                sum_vector[index] = vector_1[index] + vector_2[index];
        }
}
}  // namespace

int main(int argc, char* argv[]) try {
        const auto parsed_options = parse_options(argc, argv);
        const auto vector_size = parsed_options.vector_size;
        const auto block_size = parsed_options.block_size;

        auto [vector_1, vector_2, sum_vector] =
            create_vectors<float, 3>(vector_size);
        initialize_inputs(vector_1, vector_2, vector_size);

        const auto bytes = vector_size * sizeof(float);
        cuda_buffer device_vector_1{bytes};
        cuda_buffer device_vector_2{bytes};
        cuda_buffer device_sum_vector{bytes};

        CHECK_CUDA(cudaMemcpy(device_vector_1.get(), vector_1.data(), bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(device_vector_2.get(), vector_2.data(), bytes,
                              cudaMemcpyHostToDevice));

        cuda_event_timer timer;
        const auto grid_size = static_cast<int>(
            (vector_size + static_cast<std::size_t>(block_size) - 1) /
            block_size);

        timer.start();
        vector_addition_kernel<<<grid_size, block_size>>>(
            device_vector_1.get(), device_vector_2.get(),
            device_sum_vector.get(), vector_size);
        CHECK_CUDA(cudaGetLastError());
        timer.end();

        CHECK_CUDA(cudaMemcpy(sum_vector.data(), device_sum_vector.get(), bytes,
                              cudaMemcpyDeviceToHost));

        if (!verify_first_sum(vector_1, vector_2, sum_vector, "CUDA")) {
                return EXIT_FAILURE;
        }

        const auto seconds = timer.elapsed();
        const auto vector_sum = sum_values(sum_vector);
        const auto total_size_gb = vector_total_size_gb(vector_size);
        std::cout << core::write(cuda_vector_add_report{
                         .mode = "cuda",
                         .vector_size = vector_size,
                         .block_size = block_size,
                         .seconds = seconds,
                         .bandwidth_gbs = total_size_gb / seconds,
                         .mean = vector_sum / static_cast<double>(vector_size),
                     })
                  << '\n';

        return EXIT_SUCCESS;
} catch (const std::exception& exception) {
        core::ERROR("{}", exception.what());
        return EXIT_FAILURE;
}
