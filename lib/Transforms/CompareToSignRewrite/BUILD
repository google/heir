load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CompareToSignRewrite",
    srcs = ["CompareToSignRewrite.cpp"],
    hdrs = ["CompareToSignRewrite.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/MathExt/IR:Dialect",
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
    pass_name = "CompareToSignRewrite",
    td_file = "CompareToSignRewrite.td",
)
