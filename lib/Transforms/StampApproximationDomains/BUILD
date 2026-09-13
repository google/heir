load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "StampApproximationDomains",
    srcs = ["StampApproximationDomains.cpp"],
    hdrs = ["StampApproximationDomains.h"],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "StampApproximationDomains",
    td_file = "StampApproximationDomains.td",
)
