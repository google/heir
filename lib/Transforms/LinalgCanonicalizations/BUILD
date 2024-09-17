load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "LinalgCanonicalizations",
    srcs = ["LinalgCanonicalizations.cpp"],
    hdrs = [
        "LinalgCanonicalizations.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)
# LinalgCanonicalizations tablegen and headers.

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=LinalgCanonicalizations",
            ],
            "LinalgCanonicalizations.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "LinalgCanonicalizationsPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LinalgCanonicalizations.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
