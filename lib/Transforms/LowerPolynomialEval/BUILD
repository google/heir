load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "LowerPolynomialEval",
    srcs = ["LowerPolynomialEval.cpp"],
    hdrs = ["LowerPolynomialEval.h"],
    deps = [
        ":Patterns",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "Patterns",
    srcs = ["Patterns.cpp"],
    hdrs = ["Patterns.h"],
    deps = [
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Kernel:AbstractValue",
        "@heir//lib/Kernel:ArithmeticDag",
        "@heir//lib/Kernel:IRMaterializingVisitor",
        "@heir//lib/Utils",
        "@heir//lib/Utils/Polynomial",
        "@heir//lib/Utils/Polynomial:ChebyshevPatersonStockmeyer",
        "@heir//lib/Utils/Polynomial:Horner",
        "@heir//lib/Utils/Polynomial:PatersonStockmeyer",
        "@heir//lib/Utils/Polynomial:PolynomialTestVisitors",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "LowerPolynomialEval",
    td_file = "LowerPolynomialEval.td",
)
