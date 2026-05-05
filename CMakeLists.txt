cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(HPC LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default is Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Add CLI11
include(FetchContent)
FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(CLI11)

# Add Spdlog
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(spdlog)

# Add OpenMP
find_package(OpenMP REQUIRED)

# OpenBLAS from source, tuned for Sapphire Rapids (Emerald Rapids compatible)
set(BUILD_WITHOUT_LAPACK ON CACHE BOOL "")
set(BUILD_STATIC_LIBS ON CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(ONLY_CBLAS ON CACHE BOOL "")
set(BUILD_SINGLE ON CACHE BOOL "")
set(BUILD_DOUBLE OFF CACHE BOOL "")
set(BUILD_COMPLEX OFF CACHE BOOL "")
set(BUILD_COMPLEX16 OFF CACHE BOOL "")
set(TARGET SAPPHIRERAPIDS CACHE STRING "OpenBLAS CPU target")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_Declare(
  OpenBLAS
  GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
  GIT_TAG v0.3.28
)
FetchContent_MakeAvailable(OpenBLAS)

target_include_directories(openblas_static
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)

# Add -march=native cxxflag
option(USE_NATIVE_ARCH "Compile for the host CPU (-march=native)" ON)

# === vector addition: OpenMP implementation ===
add_executable(vector_addition-openmp src/vector_addition/vector_addition-openmp.cc)

target_compile_features(vector_addition-openmp PRIVATE cxx_std_23)

target_compile_options(vector_addition-openmp PRIVATE
    -O3
    -Wall
    -Wextra
    -ffast-math
    $<$<BOOL:${USE_NATIVE_ARCH}>:-march=native>
)

target_link_libraries(vector_addition-openmp PRIVATE OpenMP::OpenMP_CXX CLI11::CLI11 spdlog::spdlog)

# === vector addition: CUDA implementation ===
add_executable(vector_addition-cuda src/vector_addition/vector_addition-cuda.cu)

set_target_properties(vector_addition-cuda PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

target_compile_options(vector_addition-cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_link_libraries(vector_addition-cuda PRIVATE CLI11::CLI11 spdlog::spdlog)

# === matrix multiplication: hand-written OpenMP implementation
add_executable(matrix_multiplication-handwritten src/matrix_multiplication/matrix_multiplication-handwritten.cc)

target_compile_features(matrix_multiplication-handwritten PRIVATE cxx_std_23)

target_compile_options(matrix_multiplication-handwritten PRIVATE
    -O3
    -Wall
    -Wextra
    -ffast-math
    $<$<BOOL:${USE_NATIVE_ARCH}>:-march=native>
)

target_link_libraries(matrix_multiplication-handwritten PRIVATE OpenMP::OpenMP_CXX CLI11::CLI11 spdlog::spdlog)

# === matrix multiplication: OpenBLAS implementation
add_executable(matrix_multiplication-openblas src/matrix_multiplication/matrix_multiplication-openblas.cc)

target_compile_features(matrix_multiplication-openblas PRIVATE cxx_std_23)

target_compile_options(matrix_multiplication-openblas PRIVATE
    -O3
    -Wall
    -Wextra
    -ffast-math
    $<$<BOOL:${USE_NATIVE_ARCH}>:-march=native>
)

target_link_libraries(matrix_multiplication-openblas PRIVATE CLI11::CLI11 spdlog::spdlog openblas_static)

# === 2D stencil templates ===
add_executable(stencil_2d-baseline src/stencil_2d/stencil_2d-baseline.cc)

target_compile_features(stencil_2d-baseline PRIVATE cxx_std_23)

target_compile_options(stencil_2d-baseline PRIVATE
    -O3
    -Wall
    -Wextra
    -ffast-math
    $<$<BOOL:${USE_NATIVE_ARCH}>:-march=native>
)

target_link_libraries(stencil_2d-baseline PRIVATE CLI11::CLI11 spdlog::spdlog)

add_executable(stencil_2d-openmp src/stencil_2d/stencil_2d-openmp.cc)

target_compile_features(stencil_2d-openmp PRIVATE cxx_std_23)

target_compile_options(stencil_2d-openmp PRIVATE
    -O3
    -Wall
    -Wextra
    -ffast-math
    $<$<BOOL:${USE_NATIVE_ARCH}>:-march=native>
)

target_link_libraries(stencil_2d-openmp PRIVATE OpenMP::OpenMP_CXX CLI11::CLI11 spdlog::spdlog)

add_executable(stencil_2d-tiled src/stencil_2d/stencil_2d-tiled.cc)

target_compile_features(stencil_2d-tiled PRIVATE cxx_std_23)

target_compile_options(stencil_2d-tiled PRIVATE
    -O3
    -Wall
    -Wextra
    -ffast-math
    $<$<BOOL:${USE_NATIVE_ARCH}>:-march=native>
)

target_link_libraries(stencil_2d-tiled PRIVATE OpenMP::OpenMP_CXX CLI11::CLI11 spdlog::spdlog)

# === CSR SpMV templates ===
add_executable(spmv_csr-baseline src/spmv_csr/spmv_csr-baseline.cc)

target_compile_features(spmv_csr-baseline PRIVATE cxx_std_23)

target_compile_options(spmv_csr-baseline PRIVATE
    -O3
    -Wall
    -Wextra
    -ffast-math
    $<$<BOOL:${USE_NATIVE_ARCH}>:-march=native>
)

target_link_libraries(spmv_csr-baseline PRIVATE CLI11::CLI11 spdlog::spdlog)

add_executable(spmv_csr-openmp src/spmv_csr/spmv_csr-openmp.cc)

target_compile_features(spmv_csr-openmp PRIVATE cxx_std_23)

target_compile_options(spmv_csr-openmp PRIVATE
    -O3
    -Wall
    -Wextra
    -ffast-math
    $<$<BOOL:${USE_NATIVE_ARCH}>:-march=native>
)

target_link_libraries(spmv_csr-openmp PRIVATE OpenMP::OpenMP_CXX CLI11::CLI11 spdlog::spdlog)

add_executable(spmv_csr-balanced src/spmv_csr/spmv_csr-balanced.cc)

target_compile_features(spmv_csr-balanced PRIVATE cxx_std_23)

target_compile_options(spmv_csr-balanced PRIVATE
    -O3
    -Wall
    -Wextra
    -ffast-math
    $<$<BOOL:${USE_NATIVE_ARCH}>:-march=native>
)

target_link_libraries(spmv_csr-balanced PRIVATE OpenMP::OpenMP_CXX CLI11::CLI11 spdlog::spdlog)
