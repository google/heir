load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "MathToEmitC",
    srcs = ["MathToEmitC.cpp"],
    hdrs = ["MathToEmitC.h"],
    deps = [
        "@llvm-project//mlir:ConvertToEmitCInterface",
        "@llvm-project//mlir:EmitCDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)
