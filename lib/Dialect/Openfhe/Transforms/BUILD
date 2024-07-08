load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

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
        ":ConfigureCryptoContext",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
    ],
)

cc_library(
    name = "ConfigureCryptoContext",
    srcs = ["ConfigureCryptoContext.cpp"],
    hdrs = [
        "ConfigureCryptoContext.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/MulDepthAnalysis",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Openfhe",
            ],
            "Passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "OpenfhePasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
