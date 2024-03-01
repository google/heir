# FullLoopUnroll tablegen and headers.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files([
    "FullLoopUnroll.h",
])

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=FullLoopUnroll",
            ],
            "FullLoopUnroll.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "FullLoopUnrollPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FullLoopUnroll.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
