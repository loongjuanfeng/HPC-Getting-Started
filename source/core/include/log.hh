#include <spdlog/spdlog.h>

#include <utility>

namespace core {
template <typename... Arguments>
void INFO(Arguments&&... arguments) {
        spdlog::info(std::forward<Arguments>(arguments)...);
}

template <typename... Arguments>
void WARN(Arguments&&... arguments) {
        spdlog::warn(std::forward<Arguments>(arguments)...);
}

template <typename... Arguments>
void ERROR(Arguments&&... arguments) {
        spdlog::error(std::forward<Arguments>(arguments)...);
}
}  // namespace core
