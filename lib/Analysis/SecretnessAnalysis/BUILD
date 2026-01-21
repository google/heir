load("@rules_cc//cc:cc_library.bzl", "cc_library")

# SecretnessAnalysis analysis class
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "SecretnessAnalysis",
    srcs = ["SecretnessAnalysis.cpp"],
    hdrs = ["SecretnessAnalysis.h"],
    deps = [
        "@heir//lib/Analysis:Utils",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
