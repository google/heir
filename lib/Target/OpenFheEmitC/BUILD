load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OpenFheEmitCEmitter",
    srcs = ["OpenFheEmitCEmitter.cpp"],
    hdrs = ["OpenFheEmitCEmitter.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ConvertToEmitC",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TargetCpp",
    ],
)

cc_library(
    name = "OpenFhePybindEmitter",
    srcs = ["OpenFhePybindEmitter.cpp"],
    hdrs = ["OpenFhePybindEmitter.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:EmitCDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TargetCpp",
    ],
)

cc_library(
    name = "OpenFheEmitCRegistration",
    srcs = ["OpenFheEmitCTranslateRegistration.cpp"],
    hdrs = ["OpenFheEmitCTranslateRegistration.h"],
    deps = [
        ":OpenFheEmitCEmitter",
        ":OpenFhePybindEmitter",
        "@heir//lib/Dialect/Openfhe/Conversions/OpenFHEToEmitC",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:TensorExtAttributes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToEmitC",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:EmitCDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToEmitC",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToEmitC",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToEmitC",
        "@llvm-project//mlir:TranslateLib",
    ],
)
