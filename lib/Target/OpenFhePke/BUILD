# OpenFhePke Emitter

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OpenFheUtils",
    srcs = ["OpenFheUtils.cpp"],
    hdrs = [
        "OpenFheUtils.h",
    ],
    deps = [
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "OpenFhePkeEmitter",
    srcs = ["OpenFhePkeEmitter.cpp"],
    hdrs = [
        "OpenFhePkeEmitter.h",
        "OpenFhePkeTemplates.h",
    ],
    deps = [
        ":OpenFheUtils",
        "@heir//lib/Analysis/SelectVariableNames",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Target:Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:PolynomialDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TranslateLib",
    ],
)

cc_library(
    name = "OpenFhePkeHeaderEmitter",
    srcs = ["OpenFhePkeHeaderEmitter.cpp"],
    hdrs = [
        "OpenFhePkeHeaderEmitter.h",
        "OpenFhePkeTemplates.h",
    ],
    deps = [
        ":OpenFheUtils",
        "@heir//lib/Analysis/SelectVariableNames",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Target:Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:PolynomialDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)
