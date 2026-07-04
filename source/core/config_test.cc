#include "config.hh"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

namespace {
template <typename Type>
bool expect_equal(const Type& actual, const Type& expected, const char* message) {
        if (actual == expected) {
                return true;
        }

        std::cerr << message << '\n';
        return false;
}

template <typename Type>
bool expect_optional(
    const std::optional<Type>& actual, const Type& expected,
    const char* message) {
        if (actual && *actual == expected) {
                return true;
        }

        std::cerr << message << '\n';
        return false;
}
}  // namespace

int main() {
        const auto config_path =
            std::filesystem::temp_directory_path() / "hpc_config_test.toml";
        {
                std::ofstream config{config_path};
                config << R"(
[kernel]
name = "spmv"
repeat = 5

[kernel.nested]
enabled = true
)";
        }

        const auto parsed = core::read(config_path);
        std::filesystem::remove(config_path);

        auto ok = true;
        ok &= expect_optional(
            core::get<std::string>(parsed, "kernel.name"), std::string{"spmv"},
            "read should parse string values from a file");
        ok &= expect_optional(
            core::get<int64_t>(parsed, "kernel.repeat"), int64_t{5},
            "read should parse integer values from a file");
        ok &= expect_optional(
            core::get<bool>(parsed, "kernel.nested.enabled"), true,
            "read should parse nested table values from a file");

        auto base = toml::parse(R"(
title = "base"
items = [1, 2]

[solver]
threads = 4
mode = "serial"

[solver.limits]
memory_gb = 8
time_seconds = 30
)");

        const auto custom = toml::parse(R"(
items = [3]

[solver]
mode = "openmp"

[solver.limits]
time_seconds = 60
)");

        const auto merged = core::merge(base, custom);

        ok &= expect_optional(
            core::get<std::string>(merged, "title"), std::string{"base"},
            "merge should preserve top-level base values");
        ok &= expect_optional(
            core::get<int64_t>(merged, "solver.threads"), int64_t{4},
            "merge should preserve untouched nested values");
        ok &= expect_optional(
            core::get<std::string>(merged, "solver.mode"), std::string{"openmp"},
            "merge should override nested scalar values");
        ok &= expect_optional(
            core::get<int64_t>(merged, "solver.limits.memory_gb"), int64_t{8},
            "merge should preserve deeply nested values");
        ok &= expect_optional(
            core::get<int64_t>(merged, "solver.limits.time_seconds"), int64_t{60},
            "merge should override deeply nested values");
        ok &= expect_equal(
            merged["items"].as_array()->size(), std::size_t{1},
            "merge should replace arrays wholesale");
        ok &= expect_optional(
            core::get<int64_t>(merged, "items[0]"), int64_t{3},
            "get should support toml++ path syntax for arrays");
        ok &= expect_equal(
            core::get<int64_t>(merged, "solver.missing").has_value(), false,
            "get should return empty optional for missing paths");
        ok &= expect_equal(
            core::get<int64_t>(merged, "solver.mode").has_value(), false,
            "get should return empty optional for type mismatches");
        ok &= expect_equal(
            core::get<int64_t>(merged, "solver.missing", int64_t{99}), int64_t{99},
            "get fallback overload should return the fallback when absent");

        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
