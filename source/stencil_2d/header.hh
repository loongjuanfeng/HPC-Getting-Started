#pragma once

#include <array>
#include <cstddef>
#include <numeric>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include "allocator.hh"
#include "report.hh"

namespace config {
constexpr bool use_no_initialization_allocator = true;
constexpr std::size_t default_grid_size = 8192;
constexpr std::size_t default_iterations = 100;
constexpr std::size_t default_tile_size = 256;
}  // namespace config

template <typename Type>
using grid_vector = std::conditional_t<
    config::use_no_initialization_allocator,
    std::vector<Type, core::allocator_without_initialization<Type>>,
    std::vector<Type>>;

template <typename Type, std::size_t grid_count>
auto create_grids(const std::size_t grid_size) {
        std::array<grid_vector<Type>, grid_count> grids;
        for (auto& grid : grids) {
                grid = grid_vector<Type>(grid_size * grid_size);
        }
        return grids;
}

[[nodiscard]] inline float heat_source_value(const std::size_t row,
                                             const std::size_t column,
                                             const std::size_t grid_size) {
        if (row == 0 || column == 0 || row + 1 == grid_size ||
            column + 1 == grid_size) {
                return 0.0F;
        }

        const auto lower = grid_size / 3;
        const auto upper = (2 * grid_size) / 3;
        return row >= lower && row <= upper && column >= lower &&
                       column <= upper
                   ? 1.0F
                   : 0.0F;
}

[[nodiscard]] inline double checksum_grid(const grid_vector<float>& grid) {
        return std::accumulate(grid.begin(), grid.end(), 0.0);
}

struct stencil_metrics {
        double updates;
        double flops;
        double effective_data_size_gb;
};

[[nodiscard]] inline stencil_metrics calculate_metrics(
    const std::size_t grid_size, const std::size_t iterations) {
        constexpr double giga = 1e9;
        const auto interior =
            static_cast<double>((grid_size - 2) * (grid_size - 2));
        const auto updates = static_cast<double>(iterations) * interior;
        return {
            .updates = updates,
            .flops = updates * 4.0,
            .effective_data_size_gb =
                updates * 5.0 * static_cast<double>(sizeof(float)) / giga,
        };
}

struct stencil_report {
        static constexpr std::string_view title = "2D Stencil";

        std::string_view mode;
        std::size_t grid_size;
        std::size_t iterations;
        double seconds;
        double gupdates;
        double gflops;
        double bandwidth_gbs;
        double checksum;

        [[nodiscard]] auto fields() const {
                return std::tuple{
                    core::field{"mode", mode},
                    core::field{"grid", grid_size},
                    core::field{"iterations", iterations},
                    core::field{"time used", seconds, ".4f", "s"},
                    core::field{"updates", gupdates, ".4f", "GUpdates/s"},
                    core::field{"compute", gflops, ".4f", "GFLOPS"},
                    core::field{"bandwidth", bandwidth_gbs, ".4f", "GB/s"},
                    core::field{"checksum", checksum, ".4f"},
                };
        }
};

struct threaded_stencil_report {
        static constexpr std::string_view title = "2D Stencil";

        std::string_view mode;
        std::size_t grid_size;
        std::size_t iterations;
        int threads;
        double seconds;
        double gupdates;
        double gflops;
        double bandwidth_gbs;
        double checksum;

        [[nodiscard]] auto fields() const {
                return std::tuple{
                    core::field{"mode", mode},
                    core::field{"grid", grid_size},
                    core::field{"iterations", iterations},
                    core::field{"threads", threads},
                    core::field{"time used", seconds, ".4f", "s"},
                    core::field{"updates", gupdates, ".4f", "GUpdates/s"},
                    core::field{"compute", gflops, ".4f", "GFLOPS"},
                    core::field{"bandwidth", bandwidth_gbs, ".4f", "GB/s"},
                    core::field{"checksum", checksum, ".4f"},
                };
        }
};

struct tiled_stencil_report {
        static constexpr std::string_view title = "2D Stencil";

        std::string_view mode;
        std::size_t grid_size;
        std::size_t iterations;
        std::size_t tile_size;
        int threads;
        double seconds;
        double gupdates;
        double gflops;
        double bandwidth_gbs;
        double checksum;

        [[nodiscard]] auto fields() const {
                return std::tuple{
                    core::field{"mode", mode},
                    core::field{"grid", grid_size},
                    core::field{"iterations", iterations},
                    core::field{"tile size", tile_size},
                    core::field{"threads", threads},
                    core::field{"time used", seconds, ".4f", "s"},
                    core::field{"updates", gupdates, ".4f", "GUpdates/s"},
                    core::field{"compute", gflops, ".4f", "GFLOPS"},
                    core::field{"bandwidth", bandwidth_gbs, ".4f", "GB/s"},
                    core::field{"checksum", checksum, ".4f"},
                };
        }
};
