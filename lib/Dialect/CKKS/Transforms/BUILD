load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = ["Passes.h"],
    deps = [
        ":DecomposeKeySwitch",
        ":DecomposeRelinearize",
        ":pass_inc_gen",
    ],
)

cc_library(
    name = "DecomposeRelinearize",
    srcs = ["DecomposeRelinearize.cpp"],
    hdrs = ["DecomposeRelinearize.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "DecomposeKeySwitch",
    srcs = ["DecomposeKeySwitch.cpp"],
    hdrs = ["DecomposeKeySwitch.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@llvm-project//mlir:TensorDialect",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = {
        "Passes.h.inc": [
            "-gen-pass-decls",
            "-name=CKKS",
        ],
        "CKKSPasses.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
