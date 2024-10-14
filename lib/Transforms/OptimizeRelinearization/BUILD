load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OptimizeRelinearization",
    srcs = ["OptimizeRelinearization.cpp"],
    hdrs = [
        "OptimizeRelinearization.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/OptimizeRelinearizationAnalysis",
        "@heir//lib/Dialect:Utils",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
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
                "-name=OptimizeRelinearization",
            ],
            "OptimizeRelinearization.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "OptimizeRelinearizationPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "OptimizeRelinearization.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
