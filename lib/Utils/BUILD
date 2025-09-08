load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Utils",
    srcs = ["Utils.cpp"],
    hdrs = ["Utils.h"],
    deps = [
        ":MathUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "AffineMapUtils",
    srcs = ["AffineMapUtils.cpp"],
    hdrs = ["AffineMapUtils.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "TensorUtils",
    srcs = ["TensorUtils.cpp"],
    hdrs = ["TensorUtils.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "APIntUtils",
    srcs = ["APIntUtils.cpp"],
    hdrs = ["APIntUtils.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "AttributeUtils",
    srcs = ["AttributeUtils.cpp"],
    hdrs = ["AttributeUtils.h"],
    deps = [
        "@heir//lib/Dialect:HEIRInterfaces",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ConversionUtils",
    srcs = ["ConversionUtils.cpp"],
    hdrs = ["ConversionUtils.h"],
    deps = [
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/TfheRust/IR:Dialect",
        "@heir//lib/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "ContextAwareConversionUtils",
    srcs = ["ContextAwareConversionUtils.cpp"],
    hdrs = ["ContextAwareConversionUtils.h"],
    deps = [
        ":AttributeUtils",
        ":ContextAwareDialectConversion",
        ":ContextAwareTypeConversion",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "ContextAwareTypeConversion",
    srcs = ["ContextAwareTypeConversion.cpp"],
    hdrs = ["ContextAwareTypeConversion.h"],
    deps = [
        ":AttributeUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "ContextAwareDialectConversion",
    srcs = ["ContextAwareDialectConversion.cpp"],
    hdrs = ["ContextAwareDialectConversion.h"],
    deps = [
        ":ContextAwareTypeConversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:config",
    ],
)

cc_library(
    name = "MathUtils",
    srcs = ["MathUtils.cpp"],
    hdrs = ["MathUtils.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "LogArithmetic",
    srcs = ["LogArithmetic.cpp"],
    hdrs = ["LogArithmetic.h"],
)

cc_test(
    name = "LogArithmeticTest",
    srcs = ["LogArithmeticTest.cpp"],
    deps = [
        ":LogArithmetic",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "TargetUtils",
    srcs = ["TargetUtils.cpp"],
    hdrs = ["TargetUtils.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "TransformUtils",
    srcs = ["TransformUtils.cpp"],
    hdrs = ["TransformUtils.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
    ],
)
