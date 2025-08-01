load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "RLWEParams",
    srcs = ["RLWEParams.cpp"],
    hdrs = ["RLWEParams.h"],
    deps = [
        ":RLWESecurityParams",
        "@heir//lib/Utils",
        "@llvm-project//llvm:Support",
        "@openfhe//:core",
    ],
)

cc_library(
    name = "RLWESecurityParams",
    srcs = ["RLWESecurityParams.cpp"],
    hdrs = ["RLWESecurityParams.h"],
    deps = [
    ],
)
