package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "NoiseAnalysis",
    srcs = ["NoiseAnalysis.cpp"],
    hdrs = [
        "NoiseAnalysis.h",
    ],
    deps = [
        ":Params",
        "@heir//lib/Analysis/DimensionAnalysis",
        "@heir//lib/Analysis/LevelAnalysis",
        "@heir//lib/Analysis/NoiseAnalysis/BGV:Noise",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "Params",
    srcs = ["Params.cpp"],
    hdrs = ["Params.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)
