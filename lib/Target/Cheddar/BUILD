load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CheddarEmitter",
    srcs = ["CheddarEmitter.cpp"],
    hdrs = [
        "CheddarEmitter.h",
        "CheddarTemplates.h",
    ],
    deps = [
        "@heir//lib/Analysis/SelectVariableNames",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Utils:TargetUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TranslateLib",
    ],
)
