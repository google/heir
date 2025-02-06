package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Noise",
    srcs = ["Noise.cpp"],
    hdrs = [
        "Noise.h",
    ],
    deps = [
        "@heir//lib/Analysis/NoiseAnalysis:Params",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
