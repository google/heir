load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CGGIToSCIFRBool",
    srcs = [
        "CGGIToSCIFRBool.cpp",
    ],
    hdrs = [
        "CGGIToSCIFRBool.h",
    ],
    deps = [
        ":cggi_to_scifrbool_pass_inc_gen",
        "@heir//lib/Backend/cornami/Dialect/SCIFRBool/IR:Dialect",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "ReplaceOpWithSection",
    srcs = [
        "ReplaceOpWithSection.cpp",
    ],
    hdrs = [
        "ReplaceOpWithSection.h",
    ],
    deps = [
        ":replace_op_with_section_pass_inc_gen",
        "@heir//lib/Backend/cornami/Dialect/SCIFRBool/IR:Dialect",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "cggi_to_scifrbool_pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=CGGIToSCIFRBool",
            ],
            "CGGIToSCIFRBool.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CGGIToSCIFRBool.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "replace_op_with_section_pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=ReplaceOpWithSection",
            ],
            "ReplaceOpWithSection.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ReplaceOpWithSection.td",
    deps = [
        "@heir//lib/Backend/cornami/Dialect/SCIFRBool/IR:ops_inc_gen",
        "@heir//lib/Backend/cornami/Dialect/SCIFRBool/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
