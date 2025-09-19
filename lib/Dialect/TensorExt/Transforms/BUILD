load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = [
        "Passes.h",
    ],
    deps = [
        ":CollapseInsertionChains",
        ":FoldConvertLayoutIntoAssignLayout",
        ":ImplementRotateAndReduce",
        ":ImplementShiftNetwork",
        ":InsertRotate",
        ":RotateAndReduce",
        ":pass_inc_gen",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
    ],
)

cc_library(
    name = "InsertRotate",
    srcs = ["InsertRotate.cpp"],
    hdrs = [
        "InsertRotate.h",
    ],
    deps = [
        ":insert_rotate_inc_gen",
        ":pass_inc_gen",
        "@heir//lib/Analysis/TargetSlotAnalysis",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:canonicalize_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "CollapseInsertionChains",
    srcs = ["CollapseInsertionChains.cpp"],
    hdrs = [
        "CollapseInsertionChains.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect:Utils",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "RotateAndReduce",
    srcs = ["RotateAndReduce.cpp"],
    hdrs = [
        "RotateAndReduce.h",
    ],
    deps = [
        ":ImplementRotateAndReduce",
        ":pass_inc_gen",
        "@heir//lib/Analysis/RotationAnalysis",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "ImplementShiftNetwork",
    srcs = ["ImplementShiftNetwork.cpp"],
    hdrs = [
        "ImplementShiftNetwork.h",
    ],
    deps = [
        ":RotationGroupKernel",
        ":pass_inc_gen",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Kernel:IRMaterializingVisitor",
        "@heir//lib/Utils",
        "@heir//lib/Utils:AffineMapUtils",
        "@heir//lib/Utils/ADT:FrozenVector",
        "@heir//lib/Utils/Graph",
        "@heir//lib/Utils/Layout:Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "FoldConvertLayoutIntoAssignLayout",
    srcs = ["FoldConvertLayoutIntoAssignLayout.cpp"],
    hdrs = [
        "FoldConvertLayoutIntoAssignLayout.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Utils:AttributeUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "ImplementRotateAndReduce",
    srcs = ["ImplementRotateAndReduce.cpp"],
    hdrs = [
        "ImplementRotateAndReduce.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Kernel:ArithmeticDag",
        "@heir//lib/Kernel:IRMaterializingVisitor",
        "@heir//lib/Kernel:KernelImplementation",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "RotationGroupKernel",
    hdrs = ["RotationGroupKernel.h"],
    deps = [
        "@heir//lib/Kernel:AbstractValue",
        "@heir//lib/Kernel:ArithmeticDag",
        "@heir//lib/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "ImplementShiftNetworkTest",
    srcs = ["ImplementShiftNetworkTest.cpp"],
    deps = [
        ":ImplementShiftNetwork",
        ":RotationGroupKernel",
        "@googletest//:gtest_main",
        "@heir//lib/Kernel:AbstractValue",
        "@heir//lib/Kernel:ArithmeticDag",
        "@heir//lib/Kernel:TestingUtils",
    ],
)

# TensorExt pass tablegen and headers.

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = {
        "Passes.h.inc": [
            "-gen-pass-decls",
            "-name=TensorExt",
        ],
        "TensorExtPasses.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "insert_rotate_inc_gen",
    tbl_outs = {"InsertRotate.cpp.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "InsertRotate.td",
    deps = [
        "@heir//lib/Dialect/TensorExt/IR:ops_inc_gen",
        "@heir//lib/Dialect/TensorExt/IR:td_files",
        "@heir//lib/Utils/DRR",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)
