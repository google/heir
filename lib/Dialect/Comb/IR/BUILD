load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "CombDialect.cpp",
        "CombOps.cpp",
    ],
    hdrs = [
        "CombDialect.h",
        "CombOps.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":enum_inc_gen",
        ":ops_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "Comb.td",
        "Combinational.td",
    ],
    includes = ["../../../.."],
    deps = [
        "@heir//lib/Dialect:td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    includes = ["include"],
    tbl_outs = {
        "CombDialect.h.inc": [
            "-gen-dialect-decls",
            "-dialect=comb",
        ],
        "CombDialect.cpp.inc": [
            "-gen-dialect-defs",
            "-dialect=comb",
        ],
        "CombDialect.md": [
            "-gen-dialect-doc",
            "-dialect=comb",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Comb.td",
    deps = [
        ":td_files",
        ":type_inc_gen",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    includes = ["include"],
    tbl_outs = {
        "Comb.h.inc": [
            "-gen-op-decls",
        ],
        "Comb.cpp.inc": [
            "-gen-op-defs",
        ],
        "CombOps.md": ["-gen-op-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Comb.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "type_inc_gen",
    includes = ["include"],
    tbl_outs = {
        "CombTypes.h.inc": [
            "-gen-typedef-decls",
        ],
        "CombTypes.cpp.inc": [
            "-gen-typedef-defs",
        ],
        "CombTypes.md": ["-gen-typedef-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Comb.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "enum_inc_gen",
    includes = ["include"],
    tbl_outs = {
        "CombEnums.h.inc": [
            "-gen-enum-decls",
        ],
        "CombEnums.cpp.inc": [
            "-gen-enum-defs",
        ],
        "CombEnums.md": ["-gen-enum-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Comb.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)
