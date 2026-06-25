load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ConvPoolFusion",
    srcs = ["ConvPoolFusion.cpp"],
    hdrs = ["ConvPoolFusion.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "LinalgCanonicalizations",
    srcs = ["LinalgCanonicalizations.cpp"],
    hdrs = [
        "LinalgCanonicalizations.h",
    ],
    deps = [
        ":ConvPoolFusion",
        ":pass_inc_gen",
        "@heir//lib/Utils:TensorUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:DialectUtils",
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
    pass_name = "LinalgCanonicalizations",
)

cc_test(
    name = "ConvPoolFusionTest",
    srcs = ["ConvPoolFusionTest.cpp"],
    deps = [
        ":ConvPoolFusion",
        "@fuzztest//fuzztest",
        "@googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)
