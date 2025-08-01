load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "FrozenVector",
    hdrs = ["FrozenVector.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)
