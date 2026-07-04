#include <concepts>
#include <format>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace core {
template <typename Value>
struct field {
        std::string_view name;
        Value value;
        std::string_view format = {};
        std::string_view unit = {};
};

template <typename Value>
field(std::string_view, Value&&, std::string_view = {}, std::string_view = {})
    -> field<std::decay_t<Value>>;

template <typename Report>
concept Report_Like = requires(const Report& report) {
        { Report::title } -> std::convertible_to<std::string_view>;
        { report.fields() };
};

template <typename Value>
void append(std::string& output, const field<Value>& item) {
        auto value_format = std::string{"{:>10"};
        value_format += item.format;
        value_format += '}';

        output += std::format(
            "\n{:>12} = {}", item.name,
            std::vformat(value_format, std::make_format_args(item.value)));
        if (!item.unit.empty()) {
                output += std::format(" {}", item.unit);
        }
}

template <Report_Like Report, typename To = std::string>
To write(const Report& report)
        requires std::same_as<To, std::string>
{
        auto output =
            std::format("===== {} =====", std::string_view{Report::title});

        std::apply(
            [&output](const auto&... items) { (append(output, items), ...); },
            report.fields());

        return output;
}
}  // namespace core
