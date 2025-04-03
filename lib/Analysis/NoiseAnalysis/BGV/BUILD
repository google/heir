load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "NoiseAnalysis",
    srcs = [
        "NoiseAnalysis.cpp",
    ],
    hdrs = [
    ],
    deps = [
        ":NoiseByBoundCoeffModel",
        ":NoiseByVarianceCoeffModel",
        ":NoiseCanEmbModel",
        "@heir//lib/Analysis:Utils",
        "@heir//lib/Analysis/DimensionAnalysis",
        "@heir//lib/Analysis/LevelAnalysis",
        "@heir//lib/Analysis/NoiseAnalysis",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
    # required for gcc to link properly
    alwayslink = 1,
)

cc_library(
    name = "NoiseByBoundCoeffModel",
    srcs = [
        "NoiseByBoundCoeffModel.cpp",
    ],
    hdrs = [
        "NoiseByBoundCoeffModel.h",
    ],
    deps = [
        "@heir//lib/Analysis/NoiseAnalysis:Noise",
        "@heir//lib/Parameters/BGV:Params",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "NoiseCanEmbModel",
    srcs = [
        "NoiseCanEmbModel.cpp",
    ],
    hdrs = [
        "NoiseCanEmbModel.h",
    ],
    deps = [
        "@heir//lib/Analysis/NoiseAnalysis:Noise",
        "@heir//lib/Parameters/BGV:Params",
    ],
)

cc_library(
    name = "NoiseByVarianceCoeffModel",
    srcs = [
        "NoiseByVarianceCoeffModel.cpp",
    ],
    hdrs = [
        "NoiseByVarianceCoeffModel.h",
    ],
    deps = [
        "@heir//lib/Analysis/NoiseAnalysis:Noise",
        "@heir//lib/Parameters/BGV:Params",
        "@heir//lib/Utils:MathUtils",
    ],
)

cc_library(
    name = "NoiseBySymbolCoeffModelAnalysis",
    srcs = [
        "NoiseBySymbolCoeffModelAnalysis.cpp",
    ],
    hdrs = [
    ],
    deps = [
        ":NoiseBySymbolCoeffModel",
        "@heir//lib/Analysis:Utils",
        "@heir//lib/Analysis/DimensionAnalysis",
        "@heir//lib/Analysis/LevelAnalysis",
        "@heir//lib/Analysis/NoiseAnalysis",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
    # required for gcc to link properly
    alwayslink = 1,
)

cc_library(
    name = "NoiseBySymbolCoeffModel",
    srcs = [
        "NoiseBySymbolCoeffModel.cpp",
    ],
    hdrs = [
        "NoiseBySymbolCoeffModel.h",
    ],
    deps = [
        "@heir//lib/Analysis/NoiseAnalysis:Symbolic",
        "@heir//lib/Parameters/BGV:Params",
        "@heir//lib/Utils:MathUtils",
    ],
)
