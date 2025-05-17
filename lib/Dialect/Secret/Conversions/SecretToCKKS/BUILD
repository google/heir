load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "SecretToCKKS",
    srcs = ["SecretToCKKS.cpp"],
    hdrs = [
        "SecretToCKKS.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Mgmt/IR:MgmtOps",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Dialect/RNS/IR:RNSTypes",
        "@heir//lib/Dialect/Secret/Conversions:Patterns",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:AttributeUtils",
        "@heir//lib/Utils:ContextAwareConversionUtils",
        "@heir//lib/Utils:ContextAwareDialectConversion",
        "@heir//lib/Utils:ContextAwareTypeConversion",
        "@heir//lib/Utils/Polynomial",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = {
        "SecretToCKKS.h.inc": [
            "-gen-pass-decls",
            "-name=SecretToCKKS",
        ],
        "SecretToCKKS.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SecretToCKKS.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
