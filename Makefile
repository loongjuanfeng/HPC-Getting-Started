.PHONY: all build run-va run-mm run-mm-handwritten run-mm-openblas run-stencil run-stencil-baseline run-stencil-openmp run-stencil-tiled run-spmv run-spmv-baseline run-spmv-openmp run-spmv-balanced clean

BUILD_DIR := build

all: build

build:
	cmake -S . -B $(BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	cmake --build $(BUILD_DIR)

run-va: build
	./$(BUILD_DIR)/vector_addition

run-mm: run-mm-handwritten run-mm-openblas

run-mm-handwritten: build
	./$(BUILD_DIR)/matrix_multiplication-handwritten

run-mm-openblas: build
	./$(BUILD_DIR)/matrix_multiplication-openblas

run-stencil: run-stencil-baseline run-stencil-openmp run-stencil-tiled

run-stencil-baseline: build
	./$(BUILD_DIR)/stencil_2d-baseline

run-stencil-openmp: build
	./$(BUILD_DIR)/stencil_2d-openmp

run-stencil-tiled: build
	./$(BUILD_DIR)/stencil_2d-tiled

run-spmv: run-spmv-baseline run-spmv-openmp run-spmv-balanced

run-spmv-baseline: build
	./$(BUILD_DIR)/spmv_csr-baseline

run-spmv-openmp: build
	./$(BUILD_DIR)/spmv_csr-openmp

run-spmv-balanced: build
	./$(BUILD_DIR)/spmv_csr-balanced

clean:
	rm -rf $(BUILD_DIR) CMakeCache.txt CMakeFiles cmake_install.cmake compile_commands.json _deps generated openblas_config.h vector_addition matrix_multiplication matrix_multiplication-handwritten matrix_multiplication-openblas stencil_2d-baseline stencil_2d-openmp stencil_2d-tiled spmv_csr-baseline spmv_csr-openmp spmv_csr-balanced
