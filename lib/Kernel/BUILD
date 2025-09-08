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
    deps = [
        "@llvm-project//llvm:Support",
    ],
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
        ":AbstractValue",
        ":ArithmeticDag",
        ":KernelImplementation",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Utils",
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
    name = "AbstractValue",
    hdrs = ["AbstractValue.h"],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "KernelImplementation",
    hdrs = ["KernelImplementation.h"],
    deps = [
        ":AbstractValue",
        ":ArithmeticDag",
        ":Kernel",
        "@heir//lib/Utils:APIntUtils",
        "@heir//lib/Utils:MathUtils",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "TestingUtils",
    srcs = ["TestingUtils.cpp"],
    hdrs = ["TestingUtils.h"],
    deps = [
        ":AbstractValue",
        ":ArithmeticDag",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "RotationCountVisitor",
    srcs = ["RotationCountVisitor.cpp"],
    hdrs = ["RotationCountVisitor.h"],
    deps = [
        ":AbstractValue",
        ":ArithmeticDag",
    ],
)

cc_test(
    name = "KernelImplementationTest",
    srcs = [
        "KernelImplementationTest.cpp",
        "RotateAndReduceImplTest.cpp",
    ],
    deps = [
        ":AbstractValue",
        ":ArithmeticDag",
        ":Kernel",
        ":KernelImplementation",
        ":TestingUtils",
        "@googletest//:gtest_main",
        "@heir//lib/Utils/Layout:Evaluate",
        "@heir//lib/Utils/Layout:Utils",
        "@llvm-project//mlir:IR",
    ],
)

cc_test(
    name = "KernelCostTest",
    srcs = ["KernelCostTest.cpp"],
    deps = [
        ":AbstractValue",
        ":ArithmeticDag",
        ":Kernel",
        ":KernelImplementation",
        ":RotationCountVisitor",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "RotateAndReduceFuzzTest",
    srcs = ["RotateAndReduceFuzzTest.cpp"],
    deps = [
        ":AbstractValue",
        ":ArithmeticDag",
        ":KernelImplementation",
        ":TestingUtils",
        "@fuzztest//fuzztest",
        "@fuzztest//fuzztest:fuzztest_gtest_main",
    ],
)
