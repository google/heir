load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "PolynomialCoeffFolding",
    srcs = ["PolynomialCoeffFolding.cpp"],
    hdrs = ["PolynomialCoeffFolding.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Utils/Polynomial",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "PolynomialCoeffFolding",
    td_file = "PolynomialCoeffFolding.td",
)
