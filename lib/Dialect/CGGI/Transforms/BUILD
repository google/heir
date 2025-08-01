load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

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
        ":BooleanVectorizer",
        ":DecomposeOperations",
        ":pass_inc_gen",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
    ],
)

cc_library(
    name = "BooleanVectorizer",
    srcs = ["BooleanVectorizer.cpp"],
    hdrs = [
        "BooleanVectorizer.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Utils/Graph",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "DecomposeOperations",
    srcs = ["DecomposeOperations.cpp"],
    hdrs = [
        "DecomposeOperations.h",
    ],
    deps = [
        ":decompose_operations_inc_gen",
        ":pass_inc_gen",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "CGGI",
    td_file = "Passes.td",
)

gentbl_cc_library(
    name = "decompose_operations_inc_gen",
    tbl_outs = {"DecomposeOperations.cpp.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "DecomposeOperations.td",
    deps = [
        "@heir//lib/Dialect:td_files",
        "@heir//lib/Dialect/CGGI/IR:ops_inc_gen",
        "@heir//lib/Dialect/CGGI/IR:td_files",
        "@heir//lib/Dialect/LWE/IR:ops_inc_gen",
        "@heir//lib/Dialect/LWE/IR:td_files",
        "@heir//lib/Dialect/Polynomial/IR:td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)
