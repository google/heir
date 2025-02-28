load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "PolynomialApproximation",
    srcs = ["PolynomialApproximation.cpp"],
    hdrs = ["PolynomialApproximation.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Utils/Approximation:CaratheodoryFejer",
        "@heir//lib/Utils/Polynomial",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "PolynomialApproximation",
    td_file = "PolynomialApproximation.td",
)
