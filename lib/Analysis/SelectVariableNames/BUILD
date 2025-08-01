load("@rules_cc//cc:cc_library.bzl", "cc_library")

# SelectVariableNames analysis pass
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "SelectVariableNames",
    srcs = ["SelectVariableNames.cpp"],
    hdrs = ["SelectVariableNames.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
