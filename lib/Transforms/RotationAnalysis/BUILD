load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "PassDefsTdFiles",
    srcs = ["RotationAnalysis.td"],
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "PassesIncGen",
    tbl_outs = {
        "Passes.h.inc": [
            "-gen-pass-decls",
            "-name=RotationAnalysis",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "RotationAnalysis.td",
    deps = [":PassDefsTdFiles"],
)

cc_library(
    name = "RotationAnalysis",
    srcs = ["RotationAnalysis.cpp"],
    hdrs = ["Passes.h"],
    deps = [
        ":PassesIncGen",
        "@heir//lib/Analysis/RotationAnalysis",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
