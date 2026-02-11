load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ArithToModArith",
    srcs = ["ArithToModArith.cpp"],
    hdrs = [
        "ArithToModArith.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:TensorExtOps",
        "@heir//lib/Utils",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    header_filename = "ArithToModArith.h.inc",
    pass_name = "ArithToModArith",
    td_file = "ArithToModArith.td",
    deps = [
        "@heir//lib/Dialect/ModArith/IR:ops_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
