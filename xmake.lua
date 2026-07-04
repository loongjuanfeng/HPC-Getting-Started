add_rules("mode.debug", "mode.release")

set_languages "c++23"
add_rules "plugin.compile_commands.autoupdate"

add_requires "toml++"
add_requires "cli11"
add_requires "spdlog"
add_requires "glaze"

option "cuda"
        set_default(false)
        set_showmenu(true)
        set_description "build CUDA targets"
option_end()

option "native"
        set_default(true)
        set_showmenu(true)
        set_description "add `-native` C++ compilation flag"
option_end()

target "core"
        set_kind "static"
        add_files "source/core/*.cc"
        add_includedirs("source/core/include", { public = true })

        add_packages("toml++", { public = false })
        add_packages("spdlog", { public = false })
        add_packages("glaze", { public = false })

target "report_test"
        set_kind "binary"
        add_files "source/core/report_test.cc"
        add_deps "core"

function add_hpc_target(name, file, opts)
        opts = opts or { }

        target(name)
                set_kind "binary"
                add_files(file)
                add_deps "core"

                if opts.openmp then
                        add_deps "openmp"
                end

                if opts.openblas then
                        add_deps "openblas"
                end
end
