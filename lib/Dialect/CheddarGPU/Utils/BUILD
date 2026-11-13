load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ConstantTables",
    srcs = ["ConstantTables.cpp"],
    hdrs = ["ConstantTables.h"],
    deps = [
        "@heir//lib/Dialect/CheddarGPU/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
