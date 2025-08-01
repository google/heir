load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ModArithToArith",
    srcs = ["ModArithToArith.cpp"],
    hdrs = [
        "ModArithToArith.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/RNS/IR:RNSTypes",
        "@heir//lib/Utils:APIntUtils",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = {
        "ModArithToArith.h.inc": [
            "-gen-pass-decls",
            "-name=ModArithToArith",
        ],
        "ModArithToArith.cpp.inc": ["-gen-rewriters"],
        "ModArithToArith.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ModArithToArith.td",
    deps = [
        "@heir//lib/Dialect/ModArith/IR:ops_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:td_files",
        "@heir//lib/Utils/DRR",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
