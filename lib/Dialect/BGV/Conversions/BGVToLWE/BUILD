load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "BGVToLWE",
    srcs = ["BGVToLWE.cpp"],
    hdrs = [
        "BGVToLWE.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Patterns",
        "@heir//lib/Utils/RewriteUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

add_heir_transforms(
    header_filename = "BGVToLWE.h.inc",
    pass_name = "BGVToLWE",
    td_file = "BGVToLWE.td",
)
