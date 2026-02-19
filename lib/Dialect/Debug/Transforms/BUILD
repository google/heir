load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = [
        "Passes.h",
    ],
    deps = [
        ":ValidateNames",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Debug/IR:Dialect",
    ],
)

cc_library(
    name = "ValidateNames",
    srcs = ["ValidateNames.cpp"],
    hdrs = [
        "ValidateNames.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Debug/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "Debug",
    td_file = "Passes.td",
)
