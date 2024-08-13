load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OperationBalancer",
    srcs = ["OperationBalancer.cpp"],
    hdrs = [
        "OperationBalancer.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Secret/IR:SecretOps",
        "@heir//lib/Graph",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)
# OperationBalancer tablegen and headers.

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=OperationBalancer",
            ],
            "OperationBalancer.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "OperationBalancerPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "OperationBalancer.td",
    deps = [
        "@heir//lib/Dialect/Secret/IR:SecretOps",
        "@heir//lib/Dialect/Secret/IR:SecretPatterns",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
