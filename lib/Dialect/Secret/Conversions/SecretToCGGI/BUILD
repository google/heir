load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "SecretToCGGI",
    srcs = ["SecretToCGGI.cpp"],
    hdrs = [
        "SecretToCGGI.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/Comb/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Transforms/MemrefToArith:Utils",
        "@heir//lib/Utils:ContextAwareConversionUtils",
        "@heir//lib/Utils:ConversionUtils",
        "@heir//lib/Utils/Polynomial",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = {
        "SecretToCGGI.h.inc": [
            "-gen-pass-decls",
            "-name=SecretToCGGI",
        ],
        "SecretToCGGI.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SecretToCGGI.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
