load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ArithmeticDag",
    srcs = ["ArithmeticDag.h"],
    hdrs = ["ArithmeticDag.h"],
)

cc_test(
    name = "ArithmeticDagTest",
    srcs = ["ArithmeticDagTest.cpp"],
    deps = [
        ":ArithmeticDag",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "IRMaterializingVisitor",
    srcs = ["IRMaterializingVisitor.cpp"],
    hdrs = ["IRMaterializingVisitor.h"],
    deps = [
        ":ArithmeticDag",
        ":KernelImplementation",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Utils:MathUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "Kernel",
    srcs = ["Kernel.cpp"],
    hdrs = [
        "Kernel.h",
        "KernelName.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "KernelImplementation",
    hdrs = ["KernelImplementation.h"],
    deps = [
        ":ArithmeticDag",
        ":Kernel",
        "@heir//lib/Utils:MathUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "TestingUtils",
    srcs = ["TestingUtils.cpp"],
    hdrs = ["TestingUtils.h"],
    deps = [
        ":ArithmeticDag",
        ":KernelImplementation",
    ],
)

cc_test(
    name = "KernelImplementationTest",
    srcs = [
        "KernelImplementationTest.cpp",
        "RotateAndReduceImplTest.cpp",
    ],
    deps = [
        ":ArithmeticDag",
        ":Kernel",
        ":KernelImplementation",
        ":TestingUtils",
        "@googletest//:gtest_main",
    ],
)
