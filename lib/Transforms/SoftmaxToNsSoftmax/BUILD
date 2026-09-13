load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "SoftmaxToNsSoftmax",
    srcs = ["SoftmaxToNsSoftmax.cpp"],
    hdrs = ["SoftmaxToNsSoftmax.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/MathExt/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Transforms/PropagatePadding",
        "@heir//lib/Utils/Approximation:CaratheodoryFejer",
        "@heir//lib/Utils/Polynomial",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "SoftmaxToNsSoftmax",
    td_file = "SoftmaxToNsSoftmax.td",
)
