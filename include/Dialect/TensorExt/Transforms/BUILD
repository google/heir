# InsertRotate tablegen and headers.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorExt",
            ],
            "Passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "TensorExtPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "insert_rotate_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "InsertRotate.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "InsertRotate.td",
    deps = [
        "@heir//include/Dialect/TensorExt/IR:ops_inc_gen",
        "@heir//include/Dialect/TensorExt/IR:td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)

exports_files([
    "Passes.h",
    "CollapseInsertionChains.h",
    "InsertRotate.h",
])
