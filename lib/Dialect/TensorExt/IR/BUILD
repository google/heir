# TensorExt dialect implementation

load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "TensorExtDialect.cpp",
    ],
    hdrs = [
        "TensorExtAttributes.h",
        "TensorExtDialect.h",
        "TensorExtOps.h",
    ],
    deps = [
        ":TensorExtAttributes",
        ":TensorExtOps",
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "TensorExtAttributes",
    srcs = [
        "TensorExtAttributes.cpp",
    ],
    hdrs = [
        "TensorExtAttributes.h",
        "TensorExtDialect.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "TensorExtOps",
    srcs = [
        "TensorExtOps.cpp",
    ],
    hdrs = [
        "TensorExtAttributes.h",
        "TensorExtDialect.h",
        "TensorExtOps.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":canonicalize_inc_gen",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        "@heir//lib/Utils:AffineMapUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)
# TensorExt tablegen and headers

td_library(
    name = "td_files",
    srcs = [
        "TensorExtAttributes.td",
        "TensorExtCanonicalization.td",
        "TensorExtDialect.td",
        "TensorExtOps.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../../.."],
    deps = [
        "@heir//lib/Utils/DRR",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "TensorExt",
    kind = "dialect",
    td_file = "TensorExtDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "attributes_inc_gen",
    dialect = "TensorExt",
    kind = "attribute",
    td_file = "TensorExtAttributes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "TensorExt",
    kind = "op",
    td_file = "TensorExtOps.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "canonicalize_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "TensorExtCanonicalization.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TensorExtCanonicalization.td",
    deps = [
        ":ops_inc_gen",
        ":td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)
