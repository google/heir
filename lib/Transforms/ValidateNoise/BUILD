package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ValidateNoise",
    srcs = ["ValidateNoise.cpp"],
    hdrs = [
        "@heir//include/Transforms/ValidateNoise:ValidateNoise.h",
    ],
    deps = [
        "@heir//include/Transforms/ValidateNoise:pass_inc_gen",
        "@heir//lib/Analysis/NoisePropagation:NoisePropagationAnalysis",
        "@heir//lib/Analysis/NoisePropagation:Variance",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)
