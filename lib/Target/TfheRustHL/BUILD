# TfheRustHL Emitter

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "TfheRustHLEmitter",
    srcs = ["TfheRustHLEmitter.cpp"],
    hdrs = [
        "TfheRustHLEmitter.h",
        "TfheRustHLTemplates.h",
    ],
    deps = [
        "@heir//lib/Analysis/SelectVariableNames",
        "@heir//lib/Dialect/TfheRust/IR:Dialect",
        "@heir//lib/Target/TfheRust:Utils",
        "@heir//lib/Transforms/MemrefToArith:Utils",
        "@heir//lib/Utils:TargetUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TranslateLib",
    ],
)
