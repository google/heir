package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "NoiseCoeffModelAnalysis",
    srcs = [
        "NoiseCoeffModelAnalysis.cpp",
    ],
    hdrs = [
    ],
    deps = [
        ":Noise",
        ":NoiseByBoundCoeffModel",
        ":NoiseByVarianceCoeffModel",
        "@heir//lib/Analysis:Utils",
        "@heir//lib/Analysis/DimensionAnalysis",
        "@heir//lib/Analysis/LevelAnalysis",
        "@heir//lib/Analysis/NoiseAnalysis",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Parameters/BGV:Params",
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
        ":Noise",
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
        ":Noise",
        "@heir//lib/Parameters/BGV:Params",
    ],
)

cc_library(
    name = "Noise",
    srcs = ["Noise.cpp"],
    hdrs = [
        "Noise.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
