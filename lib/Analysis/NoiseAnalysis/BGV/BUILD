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
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
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
    ],
)
