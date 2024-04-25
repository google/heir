# OpenFhePke Emitter

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OpenFheUtils",
    srcs = ["OpenFheUtils.cpp"],
    hdrs = [
        "@heir//include/Target/OpenFhePke:OpenFheUtils.h",
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
        "OpenFhePkeTemplates.h",
        "@heir//include/Target/OpenFhePke:OpenFhePkeEmitter.h",
    ],
    deps = [
        ":OpenFheUtils",
        "@heir//lib/Analysis/SelectVariableNames",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Target:Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TranslateLib",
    ],
)

cc_library(
    name = "OpenFhePkeHeaderEmitter",
    srcs = ["OpenFhePkeHeaderEmitter.cpp"],
    hdrs = [
        "OpenFhePkeTemplates.h",
        "@heir//include/Target/OpenFhePke:OpenFhePkeHeaderEmitter.h",
    ],
    deps = [
        ":OpenFheUtils",
        "@heir//lib/Analysis/SelectVariableNames",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Target:Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)
