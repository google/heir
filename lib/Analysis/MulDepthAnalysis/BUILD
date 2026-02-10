load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "MulDepthAnalysis",
    srcs = ["MulDepthAnalysis.cpp"],
    hdrs = ["MulDepthAnalysis.h"],
    deps = [
        "@heir//lib/Analysis:Utils",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
