load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CKKSToLWE",
    srcs = ["CKKSToLWE.cpp"],
    hdrs = [
        "CKKSToLWE.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
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
    header_filename = "CKKSToLWE.h.inc",
    pass_name = "CKKSToLWE",
    td_file = "CKKSToLWE.td",
)
