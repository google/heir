load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ValidateScale",
    srcs = ["ValidateScale.cpp"],
    hdrs = ["ValidateScale.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Parameters/BGV:Params",
        "@heir//lib/Parameters/CKKS:Params",
        "@heir//lib/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "ValidateScale",
    td_file = "ValidateScale.td",
)
