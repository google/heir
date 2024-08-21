load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ConvertSecretInsertToStaticInsert",
    srcs = ["ConvertSecretInsertToStaticInsert.cpp"],
    hdrs = ["ConvertSecretInsertToStaticInsert.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=ConvertSecretInsertToStaticInsert",
            ],
            "ConvertSecretInsertToStaticInsert.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "ConvertSecretInsertToStaticInsertPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ConvertSecretInsertToStaticInsert.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
