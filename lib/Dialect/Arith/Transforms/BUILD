load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = ["Passes.h"],
    deps = [
        ":EmuWideInth",
        ":pass_inc_gen",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "EmuWideInth",
    srcs = ["EmuWideInth.cpp"],
    hdrs = ["EmuWideInth.h"],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Arith",
            ],
            "Passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "ArithPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
