load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ArithExtToArith",
    srcs = ["ArithExtToArith.cpp"],
    hdrs = [
        "ArithExtToArith.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/ArithExt/IR:Dialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=ArithExtToArith",
            ],
            "ArithExtToArith.h.inc",
        ),
        (
            ["-gen-rewriters"],
            "ArithExtToArith.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ArithExtToArith.td",
    deps = [
        "@heir//lib/DRR",
        "@heir//lib/Dialect/ArithExt/IR:ops_inc_gen",
        "@heir//lib/Dialect/ArithExt/IR:td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
