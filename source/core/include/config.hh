#pragma once

#include <toml++/toml.hpp>

#include <filesystem>
#include <optional>
#include <string_view>
#include <utility>

namespace core {
using Table = toml::table;

[[nodiscard]] Table read(const std::filesystem::path& path);
[[nodiscard]] Table merge(const Table& base, const Table& custom);

template <typename Type>
[[nodiscard]] std::optional<Type> get(
    const Table& table, const std::string_view path) {
        return table.at_path(path).template value<Type>();
}

template <typename Type>
[[nodiscard]] Type get(
    const Table& table, const std::string_view path, Type fallback) {
        return get<Type>(table, path).value_or(std::move(fallback));
}
}  // namespace core
