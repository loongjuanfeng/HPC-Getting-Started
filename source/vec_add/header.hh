#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include "allocator.hh"
#include "log.hh"
#include "report.hh"

namespace config {
constexpr bool use_no_initialization_allocator = true;
constexpr std::size_t default_vector_size = 100'000'000;
constexpr int default_block_size = 256;
}  // namespace config

template <typename Type>
using vector_storage = std::conditional_t<
    config::use_no_initialization_allocator,
    std::vector<Type, core::allocator_without_initialization<Type>>,
    std::vector<Type>>;

template <typename Type, std::size_t vector_count>
auto create_vectors(const std::size_t vector_size) {
        std::array<vector_storage<Type>, vector_count> vectors;
        for (auto& vector : vectors) {
                vector = vector_storage<Type>(vector_size);
        }
        return vectors;
}

[[nodiscard]] inline double vector_total_size_gb(
    const std::size_t vector_size) {
        return 3.0 * static_cast<double>(vector_size * sizeof(float)) / 1e9;
}

[[nodiscard]] inline double sum_values(const vector_storage<float>& values) {
        return std::accumulate(values.begin(), values.end(), 0.0);
}

[[nodiscard]] inline bool verify_first_sum(
    const vector_storage<float>& vector_1,
    const vector_storage<float>& vector_2,
    const vector_storage<float>& sum_vector, const std::string_view mode) {
        const auto expected = vector_1.front() + vector_2.front();
        if (std::abs(sum_vector.front() - expected) <=
            std::numeric_limits<float>::epsilon() * 16.0F) {
                return true;
        }

        core::ERROR("{} verification failed: sum_vector[0] = {} expected {}",
                    mode, sum_vector.front(), expected);
        return false;
}

struct vector_add_report {
        static constexpr std::string_view title = "Vector Addition";

        std::string_view mode;
        std::size_t vector_size;
        double seconds;
        double bandwidth_gbs;
        double mean;

        [[nodiscard]] auto fields() const {
                return std::tuple{
                    core::field{"mode", mode},
                    core::field{"size", vector_size},
                    core::field{"time used", seconds, ".4f", "s"},
                    core::field{"bandwidth", bandwidth_gbs, ".4f", "GB/s"},
                    core::field{"mean", mean, ".4f"},
                };
        }
};

struct cuda_vector_add_report {
        static constexpr std::string_view title = "Vector Addition";

        std::string_view mode;
        std::size_t vector_size;
        int block_size;
        double seconds;
        double bandwidth_gbs;
        double mean;

        [[nodiscard]] auto fields() const {
                return std::tuple{
                    core::field{"mode", mode},
                    core::field{"size", vector_size},
                    core::field{"block size", block_size},
                    core::field{"time used", seconds, ".4f", "s"},
                    core::field{"bandwidth", bandwidth_gbs, ".4f", "GB/s"},
                    core::field{"mean", mean, ".4f"},
                };
        }
};
