load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "DropUnitDims",
    srcs = ["DropUnitDims.cpp"],
    hdrs = ["DropUnitDims.h"],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgInterfaces",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "DropUnitDims",
    td_file = "DropUnitDims.td",
)
