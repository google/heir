load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OpenFHEToEmitC",
    srcs = ["OpenfheToEmitCDialectInterface.cpp"],
    hdrs = ["OpenfheToEmitCDialectInterface.h"],
    deps = [
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Utils:TargetUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:ConvertToEmitCInterface",
        "@llvm-project//mlir:EmitCDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "open_fhe_to_emit_c",
    deprecation = "This rule probably does not need to be depended on.",
)
