#include <concepts>
#include <format>
#include <toml++/toml.hpp>
#include <glaze/glaze.hpp>

namespace core {
template <typename Report>
concept Report_Like = requires {
        { Report::title } -> std::convertible_to<std::string_view>;
};

template <Report_Like Report, typename To = std::string>
std::string write(const Report& report) {
        std::string output = {};

        output += std::format("===== {} =====", Report::title);

}
}  // namespace core
