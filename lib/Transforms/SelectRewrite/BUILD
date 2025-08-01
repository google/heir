load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "SelectRewrite",
    srcs = ["SelectRewrite.cpp"],
    hdrs = ["SelectRewrite.h"],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = {
        "SelectRewrite.h.inc": [
            "-gen-pass-decls",
            "-name=SelectRewrite",
        ],
        "SelectRewritePasses.md": ["-gen-pass-doc"],
        "SelectRewrite.cpp.inc": ["-gen-rewriters"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SelectRewrite.td",
    deps = [
        "@heir//lib/Utils/DRR",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
