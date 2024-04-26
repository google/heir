load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "UnusedMemRef",
    srcs = ["UnusedMemRef.cpp"],
    hdrs = [
        "UnusedMemRef.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)
# UnusedMemRef tablegen and headers.

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=UnusedMemRef",
            ],
            "UnusedMemRef.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "UnusedMemRefPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "UnusedMemRef.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
