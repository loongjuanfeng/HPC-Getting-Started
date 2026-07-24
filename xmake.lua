add_rules("mode.debug", "mode.release")

set_languages "c++23"
add_rules "plugin.compile_commands.autoupdate"

add_requires "toml++"
add_requires "cli11"
add_requires "spdlog"
add_requires "glaze"

add_requires("openmp", { system = true })
add_requires("openblas", { system = true })

option "with_cuda"
        set_default(false)
        set_showmenu(true)
        set_description "build CUDA targets"
option_end()

if has_config("with_cuda") then
        add_requires("cuda", { system = true, configs = { utils = { "cublas" } } })
end

option "native"
        set_default(true)
        set_showmenu(true)
        set_description "add `-native` C++ compilation flag"
option_end()

target "core"
        set_kind "static"
        add_files "source/core/allocator.cc"
        add_files "source/core/config.cc"
        add_includedirs("source/core/include", { public = true })

        add_packages("toml++", { public = true })
        add_packages("spdlog", { public = false })
        add_packages("glaze", { public = false })

target "report_test"
        set_kind "binary"
        add_files "source/core/report_test.cc"
        add_deps "core"

target "config_test"
        set_kind "binary"
        add_files "source/core/config_test.cc"
        add_deps "core"

target "allocator_test"
        set_kind "binary"
        add_files "source/core/allocator_test.cc"
        add_deps "core"

target "timer_test"
        set_kind "binary"
        add_files "source/core/timer_test.cc"
        add_deps "core"

function add_hpc_target(name, file, opts)
        opts = opts or { }

        target(name)
                set_kind "binary"
                add_files(file)
                add_deps "core"
                add_packages "cli11"
                add_packages "spdlog"

                if opts.openmp then
                        add_packages "openmp"
                end

                if opts.openblas then
                        add_packages "openblas"
                end

                if opts.cuda then
                        set_languages "c++20"
                        add_cugencodes "native"
                        add_packages "cuda"
                end

                if opts.cublas then
                        add_links "cublas"
                end
end

add_hpc_target("mat_mul_openmp", "source/mat_mul/openmp.cc", { openmp = true })
add_hpc_target("mat_mul_openblas", "source/mat_mul/openblas.cc", { openblas = true })
add_hpc_target("vec_add_openmp", "source/vec_add/openmp.cc", { openmp = true })
add_hpc_target("spmv_csr_baseline", "source/spmv_csr/baseline.cc")
add_hpc_target("spmv_csr_openmp", "source/spmv_csr/openmp.cc", { openmp = true })
add_hpc_target("spmv_csr_balanced", "source/spmv_csr/balanced.cc", { openmp = true })
add_hpc_target("stencil_2d_baseline", "source/stencil_2d/baseline.cc")
add_hpc_target("stencil_2d_openmp", "source/stencil_2d/openmp.cc", { openmp = true })
add_hpc_target("stencil_2d_tiled", "source/stencil_2d/tiled.cc", { openmp = true })
if has_config("with_cuda") then
        add_hpc_target("mat_mul_cublas", "source/mat_mul/cublas.cu", { cuda = true, cublas = true })
        add_hpc_target("vec_add_cuda", "source/vec_add/cublas.cu", { cuda = true })
end
