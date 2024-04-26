load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "FullLoopUnroll",
    srcs = ["FullLoopUnroll.cpp"],
    hdrs = [
        "FullLoopUnroll.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)
# FullLoopUnroll tablegen and headers.

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
