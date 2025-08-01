load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Utils",
    hdrs = ["Utils.h"],
    deps = [
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:Support",
    ],
)
