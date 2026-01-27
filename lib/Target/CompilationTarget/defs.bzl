"""Helper macros for working with HEIR backend configs."""

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

def heir_backend_config(name, srcs):
    td_library_name = name + "_td"
    td_library(
        name = td_library_name,
        srcs = srcs,
        includes = ["../../.."],
        deps = [
            "@heir//lib/Target/CompilationTarget:td_files",
        ],
    )
    gentbl_cc_library(
        name = name,
        tbl_outs = [
            (
                ["-gen-compilation-target-registration"],
                name + ".cpp.inc",
            ),
        ],
        tblgen = "@heir//lib/Tablegen:heir-tblgen",
        td_file = srcs[0],
        deps = [
            ":" + td_library_name,
        ],
    )
