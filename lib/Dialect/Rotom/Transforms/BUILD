# Rotom dialect transforms

load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

add_heir_transforms(
    generated_target_name = "layout_assignment_pass_inc_gen",
    header_filename = "LayoutAssignment/LayoutAssignment.h.inc",
    pass_name = "LayoutAssignment",
    registration_name = "RotomLayoutAssignment",
    td_file = "LayoutAssignment/LayoutAssignment.td",
)

cc_library(
    name = "LayoutAssignment",
    srcs = ["LayoutAssignment/LayoutAssignment.cpp"],
    hdrs = ["LayoutAssignment/LayoutAssignment.h"],
    deps = [
        ":layout_assignment_pass_inc_gen",
        "@heir//lib/Dialect/Rotom/IR:Dialect",
        "@heir//lib/Dialect/Rotom/Utils:LayoutAlignment",
        "@heir//lib/Dialect/Rotom/Utils:RotomTensorExtLayoutLowering",
        "@heir//lib/Dialect/Secret/IR:SecretPatterns",
        "@heir//lib/Kernel",
        "@heir//lib/Utils:AttributeUtils",
        "@heir//lib/Utils/Layout:IslConversion",
        "@heir//lib/Utils/Layout:Utils",
        "@heir//lib/Utils:MathUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

add_heir_transforms(
    generated_target_name = "materialize_tensor_ext_layout_pass_inc_gen",
    header_filename = "MaterializeTensorExtLayout/MaterializeTensorExtLayout.h.inc",
    pass_name = "MaterializeTensorExtLayout",
    registration_name = "RotomMaterialize",
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
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:AttributeUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

add_heir_transforms(
    generated_target_name = "seed_layout_pass_inc_gen",
    header_filename = "SeedLayout/SeedLayout.h.inc",
    pass_name = "SeedLayout",
    registration_name = "RotomSeed",
    td_file = "SeedLayout/SeedLayout.td",
)

cc_library(
    name = "SeedLayout",
    srcs = ["SeedLayout/SeedLayout.cpp"],
    hdrs = ["SeedLayout/SeedLayout.h"],
    deps = [
        ":seed_layout_pass_inc_gen",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect/Rotom/IR:Dialect",
        "@heir//lib/Dialect/Rotom/Utils:RotomTensorExtLayoutLowering",
        "@heir//lib/Dialect/Secret/IR:SecretPatterns",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:AttributeUtils",
        "@heir//lib/Utils:MathUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "Transforms",
    hdrs = ["Passes.h"],
    deps = [
        ":LayoutAssignment",
        ":MaterializeTensorExtLayout",
        ":SeedLayout",
        ":layout_assignment_pass_inc_gen",
        ":materialize_tensor_ext_layout_pass_inc_gen",
        ":seed_layout_pass_inc_gen",
        "@heir//lib/Dialect/Rotom/IR:Dialect",
        "@llvm-project//mlir:Pass",
    ],
)
