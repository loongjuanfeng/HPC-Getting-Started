#pragma once

#include <array>
#include <cstddef>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include "allocator.hh"
#include "report.hh"

namespace config {
constexpr bool use_no_initialization_allocator = true;
constexpr std::size_t default_matrix_size = 4096;
constexpr double checksum_tolerance = 500.0;
constexpr std::size_t calculation_loop = 5;
}  // namespace config

template <typename Type>
using matrix_vector = std::conditional_t<
    config::use_no_initialization_allocator,
    std::vector<Type, core::allocator_without_initialization<Type>>,
    std::vector<Type>>;

template <typename Type, std::size_t matrix_count>
auto create_matrix(const std::size_t matrix_size) {
        std::array<matrix_vector<Type>, matrix_count> matrices;
        for (auto& matrix : matrices) {
                matrix = matrix_vector<Type>(matrix_size * matrix_size);
        }
        return matrices;
}

struct matrix_metrics {
        double total_data_size_gb;
        double flops;
};

[[nodiscard]] inline matrix_metrics calculate_metrics(
    const std::size_t matrix_size) {
        constexpr double giga = 1e9;
        const auto elements = static_cast<double>(matrix_size * matrix_size);
        return {
            .total_data_size_gb =
                3.0 * elements * static_cast<double>(sizeof(float)) / giga,
            .flops = 2.0 * static_cast<double>(matrix_size) *
                     static_cast<double>(matrix_size) *
                     static_cast<double>(matrix_size),
        };
}

struct matrix_report {
        static constexpr std::string_view title = "Matrix Multiplication";

        std::string_view mode;
        std::size_t matrix_size;
        double seconds;
        double gflops;
        double bandwidth_gbs;

        [[nodiscard]] auto fields() const {
                return std::tuple{
                    core::field{"mode", mode},
                    core::field{"size", matrix_size},
                    core::field{"time used", seconds, ".4f", "s"},
                    core::field{"compute", gflops, ".4f", "GFLOPS"},
                    core::field{"bandwidth", bandwidth_gbs, ".4f", "GB/s"},
                };
        }
};

struct tiled_matrix_report {
        static constexpr std::string_view title = "Matrix Multiplication";

        std::string_view mode;
        std::size_t matrix_size;
        std::size_t tile_size;
        double seconds;
        double gflops;
        double bandwidth_gbs;

        [[nodiscard]] auto fields() const {
                return std::tuple{
                    core::field{"mode", mode},
                    core::field{"size", matrix_size},
                    core::field{"tile size", tile_size},
                    core::field{"time used", seconds, ".4f", "s"},
                    core::field{"compute", gflops, ".4f", "GFLOPS"},
                    core::field{"bandwidth", bandwidth_gbs, ".4f", "GB/s"},
                };
        }
};
