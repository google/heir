load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "RotationAnalysis",
    srcs = ["RotationAnalysis.cpp"],
    hdrs = ["RotationAnalysis.h"],
    deps = [
        ":DagBuilder",
        ":RotationEvalVisitor",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Kernel:AbstractValue",
        "@heir//lib/Kernel:ArithmeticDag",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "DagBuilder",
    srcs = ["DagBuilder.cpp"],
    hdrs = ["DagBuilder.h"],
    deps = [
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Kernel:AbstractValue",
        "@heir//lib/Kernel:ArithmeticDag",
        "@heir//lib/Kernel:Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "RotationEvalVisitor",
    srcs = ["RotationEvalVisitor.cpp"],
    hdrs = ["RotationEvalVisitor.h"],
    deps = [
        "@heir//lib/Kernel:AbstractValue",
        "@heir//lib/Kernel:ArithmeticDag",
        "@heir//lib/Kernel:EvalVisitor",
        "@heir//lib/Utils:RotationUtils",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "RotationEvalVisitorTest",
    srcs = ["RotationEvalVisitorTest.cpp"],
    deps = [
        ":RotationEvalVisitor",
        "@googletest//:gtest_main",
        "@heir//lib/Kernel:AbstractValue",
        "@heir//lib/Kernel:ArithmeticDag",
        "@heir//lib/Kernel:KernelImplementation",
    ],
)
