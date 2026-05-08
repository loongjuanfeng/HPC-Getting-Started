.PHONY: all configure ensure-configured ensure-cuda-configured ensure-cuda-driver-lib build
.PHONY: run-va run-va-openmp run-va-cuda
.PHONY: run-mm run-mm-handwritten run-mm-openblas
.PHONY: run-stencil run-stencil-baseline run-stencil-openmp run-stencil-tiled
.PHONY: run-spmv run-spmv-baseline run-spmv-openmp run-spmv-balanced
.PHONY: clean

BUILD_DIR := build
NIX := nix
NIX_FLAKE := path:$(CURDIR)
NIX_DEVELOP := $(NIX) develop --no-write-lock-file $(NIX_FLAKE) -c
CUDA ?= auto
HOST_CC ?= gcc
HOST_CXX ?= g++
HOST_AR ?= ar
HOST_RANLIB ?= ranlib
HOST_GCC_AR ?= gcc-ar
HOST_GCC_RANLIB ?= gcc-ranlib
CUDA_NVCC_HOST_FLAGS := -Xcompiler=-U_GNU_SOURCE,-D_DEFAULT_SOURCE,-D_LARGEFILE64_SOURCE
CUDA_NVCC_FLAGS := -ccbin=$(HOST_CXX) $(CUDA_NVCC_HOST_FLAGS)
CUDA_DRIVER_LIB ?= $(firstword $(wildcard \
	/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
	/usr/lib64/libcuda.so.1 \
	/usr/lib/wsl/lib/libcuda.so.1 \
	/run/opengl-driver/lib/libcuda.so.1 \
	/run/nvidia/driver/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
))
CUDA_DRIVER_LIB_DIR := $(BUILD_DIR)/cuda-driver-lib
CUDA_DRIVER_LIBRARY_PATH := $(abspath $(CUDA_DRIVER_LIB_DIR)):$${LD_LIBRARY_PATH}

VECTOR_OPENMP_TARGET := vector_addition-openmp
VECTOR_CUDA_TARGET := vector_addition-cuda
MATRIX_TARGETS := matrix_multiplication-handwritten matrix_multiplication-openblas
STENCIL_TARGETS := stencil_2d-baseline stencil_2d-openmp stencil_2d-tiled
SPMV_TARGETS := spmv_csr-baseline spmv_csr-openmp spmv_csr-balanced
CPU_TARGETS := $(VECTOR_OPENMP_TARGET) $(MATRIX_TARGETS) $(STENCIL_TARGETS) $(SPMV_TARGETS)

CLEAN_PATHS := \
	$(BUILD_DIR) \
	CMakeCache.txt \
	CMakeFiles \
	cmake_install.cmake \
	compile_commands.json \
	_deps \
	generated \
	openblas_config.h \
	vector_addition \
	$(CPU_TARGETS) \
	$(VECTOR_CUDA_TARGET) \
	matrix_multiplication

define build_targets
$(NIX_DEVELOP) cmake --build $(BUILD_DIR) --target $(1)
endef

all: build

$(BUILD_DIR)/build.ninja:
	$(MAKE) configure

configure:
	@command -v $(NIX) >/dev/null 2>&1 || { echo "error: nix is required. Install Nix, then rerun 'make configure'."; exit 1; }
	@$(NIX_DEVELOP) bash -c '\
		command -v "$(HOST_CC)" >/dev/null 2>&1 || { echo "error: C compiler not found in Nix dev shell: $(HOST_CC)"; exit 1; }; \
		command -v "$(HOST_CXX)" >/dev/null 2>&1 || { echo "error: C++ compiler not found in Nix dev shell: $(HOST_CXX)"; exit 1; }; \
		command -v "$(HOST_AR)" >/dev/null 2>&1 || { echo "error: archiver not found in Nix dev shell: $(HOST_AR)"; exit 1; }; \
		command -v "$(HOST_RANLIB)" >/dev/null 2>&1 || { echo "error: ranlib not found in Nix dev shell: $(HOST_RANLIB)"; exit 1; }; \
		command -v "$(HOST_GCC_AR)" >/dev/null 2>&1 || { echo "error: gcc-ar not found in Nix dev shell: $(HOST_GCC_AR)"; exit 1; }; \
		command -v "$(HOST_GCC_RANLIB)" >/dev/null 2>&1 || { echo "error: gcc-ranlib not found in Nix dev shell: $(HOST_GCC_RANLIB)"; exit 1; }; \
		rm -f "$(BUILD_DIR)/CMakeCache.txt"; \
		rm -rf "$(BUILD_DIR)/CMakeFiles"; \
		case "$(CUDA)" in \
			auto) \
				if command -v nvcc >/dev/null 2>&1; then \
					cuda_smoke_src=$$(mktemp /tmp/hpc-nvcc-smoke.XXXXXX.cu); \
					cuda_smoke_obj=$$(mktemp /tmp/hpc-nvcc-smoke.XXXXXX.o); \
					cuda_smoke_log=$$(mktemp /tmp/hpc-nvcc-smoke.XXXXXX.log); \
					trap "rm -f $$cuda_smoke_src $$cuda_smoke_obj $$cuda_smoke_log" EXIT; \
					printf "%s\n" \
						"#include <cuda_runtime.h>" \
						"#include <random>" \
						"#include <vector>" \
						"int main() { return 0; }" > "$$cuda_smoke_src"; \
					if nvcc -std=c++20 $(CUDA_NVCC_FLAGS) -c "$$cuda_smoke_src" -o "$$cuda_smoke_obj" >"$$cuda_smoke_log" 2>&1; then \
						cuda_flag=ON; \
						echo "Configuring with CUDA enabled: $$(command -v nvcc), host $(HOST_CXX)"; \
					else \
						cuda_flag=OFF; \
						echo "Configuring CPU-only build: nvcc found but failed the project CUDA smoke compile"; \
					fi; \
				else \
					cuda_flag=OFF; \
					echo "Configuring CPU-only build: nvcc not found"; \
				fi ;; \
			ON|on|1|true|TRUE) \
				if ! command -v nvcc >/dev/null 2>&1; then \
					echo "error: CUDA=ON requested, but nvcc was not found"; \
					exit 1; \
				fi; \
				cuda_smoke_src=$$(mktemp /tmp/hpc-nvcc-smoke.XXXXXX.cu); \
				cuda_smoke_obj=$$(mktemp /tmp/hpc-nvcc-smoke.XXXXXX.o); \
				cuda_smoke_log=$$(mktemp /tmp/hpc-nvcc-smoke.XXXXXX.log); \
				trap "rm -f $$cuda_smoke_src $$cuda_smoke_obj $$cuda_smoke_log" EXIT; \
				printf "%s\n" \
					"#include <cuda_runtime.h>" \
					"#include <random>" \
					"#include <vector>" \
					"int main() { return 0; }" > "$$cuda_smoke_src"; \
				if ! nvcc -std=c++20 $(CUDA_NVCC_FLAGS) -c "$$cuda_smoke_src" -o "$$cuda_smoke_obj" >"$$cuda_smoke_log" 2>&1; then \
					echo "error: CUDA=ON requested, but nvcc cannot compile the project CUDA headers"; \
					tail -n 20 "$$cuda_smoke_log"; \
					exit 1; \
				fi; \
				cuda_flag=ON; \
				echo "Configuring with CUDA enabled by CUDA=$(CUDA)" ;; \
			OFF|off|0|false|FALSE) \
				cuda_flag=OFF; \
				echo "Configuring CPU-only build by CUDA=$(CUDA)" ;; \
			*) \
				echo "error: CUDA must be auto, ON, or OFF"; \
				exit 1 ;; \
		esac; \
		cuda_flags_arg=; \
		if test "$$cuda_flag" = ON; then \
			cuda_flags_arg="-DCMAKE_CUDA_FLAGS=$(CUDA_NVCC_HOST_FLAGS)"; \
		fi; \
		echo "Configuring C compiler: $(HOST_CC)"; \
		echo "Configuring C++ compiler: $(HOST_CXX)"; \
		CC="$(HOST_CC)" CXX="$(HOST_CXX)" cmake -S . -B "$(BUILD_DIR)" -G Ninja \
			-DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_CXX_COMPILER="$(HOST_CXX)" \
			-DCMAKE_AR="$(HOST_AR)" \
			-DCMAKE_RANLIB="$(HOST_RANLIB)" \
			-DCMAKE_CXX_COMPILER_AR="$(HOST_GCC_AR)" \
			-DCMAKE_CXX_COMPILER_RANLIB="$(HOST_GCC_RANLIB)" \
			-DCMAKE_CUDA_HOST_COMPILER="$(HOST_CXX)" \
			-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
			-DUSE_NATIVE_ARCH=OFF \
			-DHPC_ENABLE_CUDA="$$cuda_flag" \
			$$cuda_flags_arg \
	'

ensure-configured:
	@if ! test -f "$(BUILD_DIR)/build.ninja" || ! test -f "$(BUILD_DIR)/CMakeFiles/rules.ninja"; then \
		$(MAKE) configure; \
	fi

ensure-cuda-configured:
	@if ! test -f "$(BUILD_DIR)/build.ninja" || \
		! test -f "$(BUILD_DIR)/CMakeFiles/rules.ninja" || \
		! grep -q "^HPC_ENABLE_CUDA:BOOL=ON" "$(BUILD_DIR)/CMakeCache.txt" 2>/dev/null; then \
		$(MAKE) configure CUDA=ON; \
	fi

ensure-cuda-driver-lib:
	@if ! test -e "$(CUDA_DRIVER_LIB)"; then \
		echo "error: CUDA driver library not found: $(CUDA_DRIVER_LIB)"; \
		echo "Set CUDA_DRIVER_LIB=/path/to/libcuda.so.1 and rerun."; \
		exit 1; \
	fi
	@mkdir -p "$(CUDA_DRIVER_LIB_DIR)"
	@find "$(CUDA_DRIVER_LIB_DIR)" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
	@ln -s "$(CUDA_DRIVER_LIB)" "$(CUDA_DRIVER_LIB_DIR)/libcuda.so.1"

build: ensure-configured
	$(NIX_DEVELOP) cmake --build $(BUILD_DIR)

run-va: ensure-configured
	$(call build_targets,$(VECTOR_OPENMP_TARGET))
	./$(BUILD_DIR)/$(VECTOR_OPENMP_TARGET)
	@if grep -q "^HPC_ENABLE_CUDA:BOOL=ON" "$(BUILD_DIR)/CMakeCache.txt" 2>/dev/null; then \
		$(call build_targets,$(VECTOR_CUDA_TARGET)) && \
		$(MAKE) ensure-cuda-driver-lib && \
		LD_LIBRARY_PATH="$(CUDA_DRIVER_LIBRARY_PATH)" ./$(BUILD_DIR)/$(VECTOR_CUDA_TARGET); \
	else \
		echo "Skipping $(VECTOR_CUDA_TARGET): CUDA target was not built"; \
	fi

run-va-openmp: ensure-configured
	$(call build_targets,$(VECTOR_OPENMP_TARGET))
	./$(BUILD_DIR)/$(VECTOR_OPENMP_TARGET)

run-va-cuda: ensure-cuda-configured
	$(call build_targets,$(VECTOR_CUDA_TARGET))
	$(MAKE) ensure-cuda-driver-lib
	LD_LIBRARY_PATH="$(CUDA_DRIVER_LIBRARY_PATH)" ./$(BUILD_DIR)/$(VECTOR_CUDA_TARGET)

run-mm: ensure-configured
	$(call build_targets,$(MATRIX_TARGETS))
	./$(BUILD_DIR)/matrix_multiplication-handwritten
	./$(BUILD_DIR)/matrix_multiplication-openblas

run-mm-handwritten: ensure-configured
	$(call build_targets,matrix_multiplication-handwritten)
	./$(BUILD_DIR)/matrix_multiplication-handwritten

run-mm-openblas: ensure-configured
	$(call build_targets,matrix_multiplication-openblas)
	./$(BUILD_DIR)/matrix_multiplication-openblas

run-stencil: ensure-configured
	$(call build_targets,$(STENCIL_TARGETS))
	./$(BUILD_DIR)/stencil_2d-baseline
	./$(BUILD_DIR)/stencil_2d-openmp
	./$(BUILD_DIR)/stencil_2d-tiled

run-stencil-baseline: ensure-configured
	$(call build_targets,stencil_2d-baseline)
	./$(BUILD_DIR)/stencil_2d-baseline

run-stencil-openmp: ensure-configured
	$(call build_targets,stencil_2d-openmp)
	./$(BUILD_DIR)/stencil_2d-openmp

run-stencil-tiled: ensure-configured
	$(call build_targets,stencil_2d-tiled)
	./$(BUILD_DIR)/stencil_2d-tiled

run-spmv: ensure-configured
	$(call build_targets,$(SPMV_TARGETS))
	./$(BUILD_DIR)/spmv_csr-baseline
	./$(BUILD_DIR)/spmv_csr-openmp
	./$(BUILD_DIR)/spmv_csr-balanced

run-spmv-baseline: ensure-configured
	$(call build_targets,spmv_csr-baseline)
	./$(BUILD_DIR)/spmv_csr-baseline

run-spmv-openmp: ensure-configured
	$(call build_targets,spmv_csr-openmp)
	./$(BUILD_DIR)/spmv_csr-openmp

run-spmv-balanced: ensure-configured
	$(call build_targets,spmv_csr-balanced)
	./$(BUILD_DIR)/spmv_csr-balanced

clean:
	rm -rf $(CLEAN_PATHS)
