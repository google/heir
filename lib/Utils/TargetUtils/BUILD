# Utilities for code generators

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "TargetUtils",
    srcs = ["TargetUtils.cpp"],
    hdrs = ["TargetUtils.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
