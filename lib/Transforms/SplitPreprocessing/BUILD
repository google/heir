load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "SplitPreprocessing",
    srcs = ["SplitPreprocessing.cpp"],
    hdrs = ["SplitPreprocessing.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Preprocessing/IR:Dialect",
        "@heir//lib/Utils:AttributeUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "SplitPreprocessing",
    td_file = "SplitPreprocessing.td",
)
