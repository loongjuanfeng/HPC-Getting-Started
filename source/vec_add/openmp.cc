#include <omp.h>

#include <CLI/CLI.hpp>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>

#include "header.hh"
#include "timer.hh"

namespace {
void setup_threads();

void initialize_inputs(vector_storage<float>& vector_1,
                       vector_storage<float>& vector_2,
                       const std::size_t vector_size) {
#pragma omp parallel for
        for (std::size_t i = 0; i < vector_size; i++) {
                thread_local std::mt19937 random_number_generator(
                    std::random_device{}());
                thread_local std::normal_distribution<float>
                    normal_distribution{1.0};
                vector_1[i] = normal_distribution(random_number_generator);
                vector_2[i] = normal_distribution(random_number_generator);
        }
}

void add_vectors(const vector_storage<float>& vector_1,
                 const vector_storage<float>& vector_2,
                 vector_storage<float>& sum_vector,
                 const std::size_t vector_size) {
#pragma omp parallel for
        for (std::size_t i = 0; i < vector_size; i++) {
                sum_vector[i] = vector_1[i] + vector_2[i];
        }
}

void setup_threads() {
        const auto* const omp_num_threads = std::getenv("OMP_NUM_THREADS");
        const auto threads_count =
            static_cast<bool>(omp_num_threads)
                ? std::atoi(omp_num_threads)
                : static_cast<int>(std::thread::hardware_concurrency());
        omp_set_num_threads(threads_count);
        core::INFO("threads = {}", threads_count);
}
}  // namespace

int main(int argc, char* argv[]) {
        CLI::App app{"OpenMP vector addition - HPC starter"};
        std::size_t vector_size{config::default_vector_size};
        app.add_option("-s,--size", vector_size, "Number of elements")
            ->check(CLI::PositiveNumber);
        CLI11_PARSE(app, argc, argv);

        setup_threads();

        auto [vector_1, vector_2, sum_vector] =
            create_vectors<float, 3>(vector_size);
        initialize_inputs(vector_1, vector_2, vector_size);

        core::Timer timer;
        timer.start();
        add_vectors(vector_1, vector_2, sum_vector, vector_size);
        timer.end();

        if (!verify_first_sum(vector_1, vector_2, sum_vector, "OpenMP")) {
                return EXIT_FAILURE;
        }

        const auto seconds = timer.elapsed();
        const auto vector_sum = sum_values(sum_vector);
        const auto total_size_gb = vector_total_size_gb(vector_size);
        std::cout << core::write(vector_add_report{
                         .mode = "openmp",
                         .vector_size = vector_size,
                         .seconds = seconds,
                         .bandwidth_gbs = total_size_gb / seconds,
                         .mean = vector_sum / static_cast<double>(vector_size),
                     })
                  << '\n';

        return EXIT_SUCCESS;
}
