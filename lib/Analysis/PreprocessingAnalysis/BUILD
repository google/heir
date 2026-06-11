load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "PreprocessingAnalysis",
    srcs = ["PreprocessingAnalysis.cpp"],
    hdrs = ["PreprocessingAnalysis.h"],
    deps = [
        "@heir//lib/Dialect:HEIRInterfaces",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
