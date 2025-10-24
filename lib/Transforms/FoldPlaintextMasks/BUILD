load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "FoldPlaintextMasks",
    srcs = ["FoldPlaintextMasks.cpp"],
    hdrs = ["FoldPlaintextMasks.h"],
    deps = [
        ":pass_inc_gen",
        ":patterns_inc_gen",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "patterns_inc_gen",
    tbl_outs = {
        "Patterns.cpp.inc": ["-gen-rewriters"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Patterns.td",
    deps = [
        "@heir//lib/Utils/DRR",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "FoldPlaintextMasks",
    td_file = "FoldPlaintextMasks.td",
)
