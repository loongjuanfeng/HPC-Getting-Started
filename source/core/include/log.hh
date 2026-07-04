#include <spdlog/spdlog.h>

#include <utility>

namespace core {
template <typename... Arguments>
void INFO(spdlog::format_string_t<Arguments...> format,
          Arguments&&... arguments) {
        spdlog::info(format, std::forward<Arguments>(arguments)...);
}

template <typename... Arguments>
void WARN(spdlog::format_string_t<Arguments...> format,
          Arguments&&... arguments) {
        spdlog::warn(format, std::forward<Arguments>(arguments)...);
}

template <typename... Arguments>
void ERROR(spdlog::format_string_t<Arguments...> format,
           Arguments&&... arguments) {
        spdlog::error(format, std::forward<Arguments>(arguments)...);
}
}  // namespace core
