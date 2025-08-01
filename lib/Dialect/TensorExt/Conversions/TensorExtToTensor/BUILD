load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "TensorExtToTensor",
    srcs = ["TensorExtToTensor.cpp"],
    hdrs = ["TensorExtToTensor.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    header_filename = "TensorExtToTensor.h.inc",
    pass_name = "TensorExtToTensor",
    td_file = "TensorExtToTensor.td",
)
