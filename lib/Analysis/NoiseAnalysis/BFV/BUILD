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
        ":NoiseByBoundCoeffModel",
        "@heir//lib/Analysis:Utils",
        "@heir//lib/Analysis/DimensionAnalysis",
        "@heir//lib/Analysis/LevelAnalysis",
        "@heir//lib/Analysis/NoiseAnalysis",
        "@heir//lib/Analysis/NoiseAnalysis:Noise",
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
        "@heir//lib/Analysis/NoiseAnalysis:Noise",
        "@heir//lib/Parameters/BGV:Params",
    ],
)
