load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

add_heir_transforms(
    header_filename = "SoftmaxCanonicalizations.h.inc",
    pass_name = "SoftmaxCanonicalizations",
    td_file = "SoftmaxCanonicalizations.td",
)

cc_library(
    name = "SoftmaxCanonicalizations",
    srcs = ["SoftmaxCanonicalizations.cpp"],
    hdrs = ["SoftmaxCanonicalizations.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/MathExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)
