#pragma once

#include <cstddef>
#include <string_view>
#include <tuple>

#include "report.hh"

struct spmv_csr_report {
        static constexpr std::string_view title = "CSR SpMV";

        std::string_view mode;
        std::string_view matrix;
        std::size_t rows;
        std::size_t nonzeros;
        double nonzeros_per_row;
        std::size_t repeat;
        double seconds;
        double throughput_gnnzs;
        double bandwidth_gbs;
        double checksum;

        [[nodiscard]] auto fields() const {
                return std::tuple{
                    core::field{"mode", mode},
                    core::field{"matrix", matrix},
                    core::field{"rows", rows},
                    core::field{"nonzeros", nonzeros},
                    core::field{"nnz/row", nonzeros_per_row, ".2f"},
                    core::field{"repeat", repeat},
                    core::field{"time used", seconds, ".4f", "s"},
                    core::field{"throughput", throughput_gnnzs, ".4f",
                                "GNNZ/s"},
                    core::field{"bandwidth", bandwidth_gbs, ".4f", "GB/s"},
                    core::field{"checksum", checksum, ".4f"},
                };
        }
};

struct threaded_spmv_csr_report {
        static constexpr std::string_view title = "CSR SpMV";

        std::string_view mode;
        std::string_view matrix;
        std::size_t rows;
        std::size_t nonzeros;
        double nonzeros_per_row;
        int threads;
        std::size_t repeat;
        double seconds;
        double throughput_gnnzs;
        double bandwidth_gbs;
        double checksum;

        [[nodiscard]] auto fields() const {
                return std::tuple{
                    core::field{"mode", mode},
                    core::field{"matrix", matrix},
                    core::field{"rows", rows},
                    core::field{"nonzeros", nonzeros},
                    core::field{"nnz/row", nonzeros_per_row, ".2f"},
                    core::field{"threads", threads},
                    core::field{"repeat", repeat},
                    core::field{"time used", seconds, ".4f", "s"},
                    core::field{"throughput", throughput_gnnzs, ".4f",
                                "GNNZ/s"},
                    core::field{"bandwidth", bandwidth_gbs, ".4f", "GB/s"},
                    core::field{"checksum", checksum, ".4f"},
                };
        }
};
