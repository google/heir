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
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
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

cc_library(
    name = "Symbolic",
    srcs = ["Symbolic.cpp"],
    hdrs = [
        "Symbolic.h",
    ],
    deps = [
        ":Noise",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)
