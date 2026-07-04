#include "report.hh"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>

namespace {
struct demo_report {
        static constexpr std::string_view title = "Demo Report";

        int size = 42;
        double seconds = 1.25;
        std::string mode = "openmp";

        [[nodiscard]] auto fields() const {
                return std::tuple{
                    core::field{"size", size},
                    core::field{"time used", seconds, ".4f", "s"},
                    core::field{"mode", mode},
                };
        }
};
}  // namespace

int main() {
        const auto output = core::write(demo_report{});
        const std::string expected =
            "===== Demo Report =====\n"
            "        size =         42\n"
            "   time used =     1.2500 s\n"
            "        mode =     openmp";

        if (output != expected) {
                std::cerr << "Expected:\n"
                          << expected << "\n\nActual:\n"
                          << output << '\n';
                return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
}
