cmake_minimum_required(VERSION 3.28)

project(HPC LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default is Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(HPC_ENABLE_CUDA "Build CUDA benchmark targets" ON)
option(USE_NATIVE_ARCH "Compile for the host CPU (-march=native)" ON)

if(HPC_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()

    enable_language(CUDA)
endif()

find_package(CLI11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(OpenMP REQUIRED)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(OPENBLAS QUIET IMPORTED_TARGET openblas)
endif()

if(TARGET PkgConfig::OPENBLAS)
    add_library(OpenBLAS::OpenBLAS ALIAS PkgConfig::OPENBLAS)
else()
    find_path(OPENBLAS_INCLUDE_DIR
        NAMES cblas.h
        PATH_SUFFIXES include openblas include/openblas
    )
    find_library(OPENBLAS_LIBRARY NAMES openblas)

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(OpenBLAS
        REQUIRED_VARS OPENBLAS_INCLUDE_DIR OPENBLAS_LIBRARY
    )

    add_library(OpenBLAS::OpenBLAS UNKNOWN IMPORTED)
    set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
        IMPORTED_LOCATION "${OPENBLAS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENBLAS_INCLUDE_DIR}"
    )
endif()

add_library(hpc_cxx_options INTERFACE)
target_compile_features(hpc_cxx_options INTERFACE cxx_std_23)
target_compile_options(hpc_cxx_options INTERFACE
    -O3
    -Wall
    -Wextra
    -ffast-math
    $<$<BOOL:${USE_NATIVE_ARCH}>:-march=native>
)

add_library(hpc_common_libs INTERFACE)
target_link_libraries(hpc_common_libs INTERFACE CLI11::CLI11 spdlog::spdlog)

function(hpc_add_cpp_executable target source)
    cmake_parse_arguments(HPC_TARGET "OPENMP;OPENBLAS" "" "" ${ARGN})

    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE hpc_cxx_options hpc_common_libs)

    if(HPC_TARGET_OPENMP)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(HPC_TARGET_OPENBLAS)
        target_link_libraries(${target} PRIVATE OpenBLAS::OpenBLAS)
    endif()
endfunction()

# === vector addition ===
hpc_add_cpp_executable(vector_addition-openmp
    src/vector_addition/vector_addition-openmp.cc
    OPENMP
)

if(HPC_ENABLE_CUDA)
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

    target_link_libraries(vector_addition-cuda PRIVATE spdlog::spdlog)
endif()

# === matrix multiplication ===
hpc_add_cpp_executable(matrix_multiplication-handwritten
    src/matrix_multiplication/matrix_multiplication-handwritten.cc
    OPENMP
)

hpc_add_cpp_executable(matrix_multiplication-openblas
    src/matrix_multiplication/matrix_multiplication-openblas.cc
    OPENBLAS
)

# === 2D stencil ===
hpc_add_cpp_executable(stencil_2d-baseline
    src/stencil_2d/stencil_2d-baseline.cc
)

hpc_add_cpp_executable(stencil_2d-openmp
    src/stencil_2d/stencil_2d-openmp.cc
    OPENMP
)

hpc_add_cpp_executable(stencil_2d-tiled
    src/stencil_2d/stencil_2d-tiled.cc
    OPENMP
)

# === CSR SpMV ===
hpc_add_cpp_executable(spmv_csr-baseline
    src/spmv_csr/spmv_csr-baseline.cc
)

hpc_add_cpp_executable(spmv_csr-openmp
    src/spmv_csr/spmv_csr-openmp.cc
    OPENMP
)

hpc_add_cpp_executable(spmv_csr-balanced
    src/spmv_csr/spmv_csr-balanced.cc
    OPENMP
)
