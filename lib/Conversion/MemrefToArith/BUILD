# MemrefToArith pass eliminates memrefs in favor of inline arithmetic constants.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Utils",
    srcs = ["Utils.cpp"],
    hdrs = ["Utils.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "UtilsTest",
    srcs = ["UtilsTest.cpp"],
    deps = [
        ":Utils",
        "@googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "MemrefGlobalReplace",
    srcs = ["MemrefGlobalReplace.cpp"],
    hdrs = ["MemrefToArith.h"],
    deps = [
        ":Utils",
        ":pass_inc_gen",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "ExpandCopy",
    srcs = ["ExpandCopy.cpp"],
    hdrs = ["MemrefToArith.h"],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "ExtractLoopBody",
    srcs = ["ExtractLoopBody.cpp"],
    hdrs = ["MemrefToArith.h"],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "UnrollAndForward",
    srcs = ["UnrollAndForward.cpp"],
    hdrs = ["MemrefToArith.h"],
    deps = [
        ":Utils",
        ":pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "MemrefToArithRegistration",
    srcs = ["PassRegistration.cpp"],
    deps = [
        ":ExpandCopy",
        ":ExtractLoopBody",
        ":MemrefGlobalReplace",
        ":UnrollAndForward",
        ":pass_inc_gen",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=MemrefToArith",
            ],
            "MemrefToArith.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "MemrefToArith.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "MemrefToArith.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
