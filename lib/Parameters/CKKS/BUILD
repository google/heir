load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Params",
    srcs = ["Params.cpp"],
    hdrs = ["Params.h"],
    deps = [
        "@heir//lib/Parameters:RLWEParams",
        "@heir//lib/Parameters:RLWESecurityParams",
        "@llvm-project//llvm:Support",
        "@openfhe//:core",
    ],
)

cc_library(
    name = "Utils",
    srcs = ["Utils.cpp"],
    hdrs = ["Utils.h"],
    deps = [
        ":Params",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Enums",
        "@heir//lib/Parameters:RLWEParams",
        "@openfhe//:core",
    ],
)
