load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ConvertSecretForToStaticFor",
    srcs = ["ConvertSecretForToStaticFor.cpp"],
    hdrs = ["ConvertSecretForToStaticFor.h"],
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
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=ConvertSecretForToStaticFor",
            ],
            "ConvertSecretForToStaticFor.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "ConvertSecretForToStaticForPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ConvertSecretForToStaticFor.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
