load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "SecretDialect.cpp",
    ],
    hdrs = [
        "SecretAttributes.h",
        "SecretDialect.h",
        "SecretOps.h",
        "SecretTypes.h",
    ],
    deps = [
        ":SecretAttributes",
        ":SecretOps",
        ":SecretPatterns",
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Kernel",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)

cc_library(
    name = "SecretPatterns",
    srcs = [
        "SecretPatterns.cpp",
    ],
    hdrs = [
        "SecretDialect.h",
        "SecretOps.h",
        "SecretPatterns.h",
        "SecretTypes.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Kernel",
        "@heir//lib/Utils:AttributeUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "SecretAttributes",
    hdrs = [
        "SecretAttributes.h",
        "SecretDialect.h",
    ],
    deps = [
        ":attributes_inc_gen",
        ":dialect_inc_gen",
        "@heir//lib/Kernel",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "SecretOps",
    srcs = [
        "SecretOps.cpp",
    ],
    hdrs = [
        "SecretDialect.h",
        "SecretOps.h",
        "SecretTypes.h",
    ],
    deps = [
        ":SecretPatterns",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Kernel",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "SecretAttributes.td",
        "SecretDialect.td",
        "SecretOps.td",
        "SecretTypes.td",
    ],
    deps = [
        "@heir//lib/Dialect:td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "Secret",
    kind = "dialect",
    td_file = "SecretDialect.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "attributes_inc_gen",
    dialect = "Secret",
    kind = "attribute",
    td_file = "SecretAttributes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "Secret",
    kind = "type",
    td_file = "SecretTypes.td",
    deps = [
        ":td_files",
    ],
)

add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "Secret",
    kind = "op",
    td_file = "SecretOps.td",
    deps = [
        ":td_files",
    ],
)
