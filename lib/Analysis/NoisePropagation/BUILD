# NoisePropagationAnalysis analysis pass
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "NoisePropagationAnalysis",
    srcs = ["NoisePropagationAnalysis.cpp"],
    hdrs = ["@heir//include/Analysis/NoisePropagation:NoisePropagationAnalysis.h"],
    deps = [
        ":Variance",
        "@heir//lib/Interfaces:NoiseInterfaces",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "Variance",
    srcs = ["Variance.cpp"],
    hdrs = ["@heir//include/Analysis/NoisePropagation:Variance.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
