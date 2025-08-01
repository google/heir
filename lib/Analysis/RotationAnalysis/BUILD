load("@rules_cc//cc:cc_library.bzl", "cc_library")

# RotationAnalysis analysis pass
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "RotationAnalysis",
    srcs = ["RotationAnalysis.cpp"],
    hdrs = ["RotationAnalysis.h"],
    deps = [
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)
