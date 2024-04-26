load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ApplyFolders",
    srcs = ["ApplyFolders.cpp"],
    hdrs = [
        "ApplyFolders.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)
# ApplyFolders tablegen and headers.

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=ApplyFolders",
            ],
            "ApplyFolders.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "ApplyFoldersPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ApplyFolders.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
