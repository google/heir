# TfheRustBool Emitter

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "TfheRustBoolEmitter",
    srcs = ["TfheRustBoolEmitter.cpp"],
    hdrs = [
        "TfheRustBoolEmitter.h",
        "TfheRustBoolTemplates.h",
    ],
    deps = [
        "@heir//lib/Analysis/SelectVariableNames",
        "@heir//lib/Dialect/TfheRustBool/IR:Dialect",
        "@heir//lib/Target/TfheRust:Utils",
        "@heir//lib/Utils:TargetUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TranslateLib",
    ],
)
