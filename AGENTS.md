# Repository Guidelines

## Project Structure & Module Organization

This repository contains standalone C++23 HPC kernels built as separate executables. Source files live in `src/`, with one translation unit per benchmark or implementation variant, for example `vector_addition.cc`, `stencil_2d-openmp.cc`, and `spmv_csr-balanced.cc`. Build products, fetched dependencies, generated headers, and `compile_commands.json` are placed under `build/`; treat this directory as generated output. Core build configuration is in `CMakeLists.txt`, `CMakePresets.json`, and the convenience `Makefile`.

## Build, Test, and Development Commands

- `make build`: configures a Release Ninja build in `build/` and compiles all executables.
- `cmake --workflow --preset ninja-release-full`: equivalent preset-driven configure and build flow.
- `make run-va`: builds and runs `build/vector_addition`.
- `make run-mm`, `make run-stencil`, `make run-spmv`: run all variants for matrix multiplication, stencil, or CSR SpMV.
- `make clean`: removes generated build artifacts.

Dependencies are fetched by CMake with `FetchContent` and include CLI11, spdlog, and OpenBLAS. OpenMP is required from the host toolchain.

## Coding Style & Naming Conventions

Use C++23 and keep each benchmark variant in its own `src/<kernel>-<variant>.cc` file. Follow the existing executable naming pattern, such as `matrix_multiplication-openblas` or `stencil_2d-tiled`, and mirror that name in the source file. Formatting is defined by `.clang-format`: LLVM base style, 4-space indentation, attached braces, 100-column limit, and left-aligned pointers. Prefer explicit CLI options through CLI11 and structured logging through spdlog when adding user-facing runtime behavior.

## Testing Guidelines

There is no dedicated test framework or `ctest` suite in the current tree. Validate changes by building all targets with `make build` and running the affected benchmark target with a small input size where supported, for example `./build/vector_addition --size 1000000`. For numerical kernels, add or preserve lightweight correctness checks before reporting timing results.

## Commit & Pull Request Guidelines

This workspace does not include usable Git history, so no project-specific commit convention can be inferred. Use concise, imperative commit messages such as `Add tiled SpMV benchmark` or `Fix OpenMP thread setup`. Pull requests should include a short behavior summary, affected kernels, build/run commands used for verification, and any relevant performance notes or hardware assumptions.

## Agent-Specific Notes

Local Codex instructions may request prefixing shell commands with `rtk`; use it when available. Do not edit generated files in `build/` directly.
