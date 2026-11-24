load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "LWEToStableHLO",
    srcs = ["LWEToStableHLO.cpp"],
    hdrs = ["LWEToStableHLO.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Dialect/RNS/IR:Dialect",
        "@heir//lib/Utils:ConversionUtils",
        "@heir//lib/Utils/Polynomial",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
)

add_heir_transforms(
    header_filename = "LWEToStableHLO.h.inc",
    pass_name = "LWEToStableHLO",
    td_file = "LWEToStableHLO.td",
)

cc_test(
    name = "LWEToStableHLOTest",
    srcs = ["LWEToStableHLOTest.cpp"],
    deps = [
        ":LWEToStableHLO",
        "@googletest//:gtest_main",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Dialect/RNS/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@stablehlo//:reference_api",
        "@stablehlo//:reference_configuration",
        "@stablehlo//:stablehlo_ops",
    ],
)
