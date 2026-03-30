# Rotom dialect transforms

load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

add_heir_transforms(
    generated_target_name = "materialize_tensor_ext_layout_pass_inc_gen",
    header_filename = "MaterializeTensorExtLayout/MaterializeTensorExtLayout.h.inc",
    pass_name = "MaterializeTensorExtLayout",
    registration_name = "Rotom",
    td_file = "MaterializeTensorExtLayout/MaterializeTensorExtLayout.td",
)

cc_library(
    name = "MaterializeTensorExtLayout",
    srcs = ["MaterializeTensorExtLayout/MaterializeTensorExtLayout.cpp"],
    hdrs = ["MaterializeTensorExtLayout/MaterializeTensorExtLayout.h"],
    deps = [
        ":materialize_tensor_ext_layout_pass_inc_gen",
        "@heir//lib/Dialect/Rotom/IR:Dialect",
        "@heir//lib/Dialect/Rotom/Utils:RotomTensorExtLayoutLowering",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:AttributeUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "Transforms",
    hdrs = ["Passes.h"],
    deps = [
        ":MaterializeTensorExtLayout",
        "@heir//lib/Dialect/Rotom/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)
