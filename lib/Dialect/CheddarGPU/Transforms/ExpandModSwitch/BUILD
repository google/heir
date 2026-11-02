load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = {
        "ExpandModSwitch.h.inc": [
            "-gen-pass-decls",
            "-name=ExpandModSwitch",
        ],
        "ExpandModSwitch.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "ExpandModSwitch",
    srcs = ["ExpandModSwitch.cpp"],
    hdrs = ["ExpandModSwitch.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/CheddarGPU/IR:Dialect",
        "@heir//lib/Utils/CheddarRef:mod_switch_tables",
        "@heir//lib/Utils/CheddarRef:params",
        "@heir//lib/Utils/CheddarRef:tables",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)
