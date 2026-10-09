load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "NoiseAnalysis",
    srcs = [],
    hdrs = [
        "NoiseAnalysis.h",
    ],
    deps = [
        "@heir//lib/Analysis:Utils",
        "@heir//lib/Analysis/DimensionAnalysis",
        "@heir//lib/Analysis/LevelAnalysis",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "Noise",
    srcs = ["Noise.cpp"],
    hdrs = [
        "Noise.h",
    ],
    deps = [
        "@heir//lib/Utils:LogArithmetic",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
