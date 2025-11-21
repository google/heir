load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

gentbl_cc_library(
    name = "patterns_inc_gen",
    tbl_outs = {"Rewrites.cpp.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Rewrites.td",
    deps = [
        "@heir//lib/Dialect/MathExt/IR:td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:MathOpsTdFiles",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)

cc_library(
    name = "ActivationCanonicalizations",
    srcs = ["ActivationCanonicalizations.cpp"],
    hdrs = ["ActivationCanonicalizations.h"],
    deps = [
        ":pass_inc_gen",
        ":patterns_inc_gen",
        "@heir//lib/Dialect/MathExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "ActivationCanonicalizations",
    td_file = "ActivationCanonicalizations.td",
)
