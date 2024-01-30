# ForwardStoreToLoad tablegen and headers.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files([
    "ForwardStoreToLoad.h",
])

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=ForwardStoreToLoad",
            ],
            "ForwardStoreToLoad.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "ForwardStoreToLoadPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ForwardStoreToLoad.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
