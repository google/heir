# OpenFhePke Emitter

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OpenFheRegistration",
    srcs = [
        "OpenFheTranslateRegistration.cpp",
    ],
    hdrs = [
        "OpenFheTranslateRegistration.h",
    ],
    deps = [
        ":OpenFhePkeEmitter",
        ":OpenFhePkeHeaderEmitter",
        ":OpenFhePkePybindEmitter",
        ":OpenFheUtils",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Dialect/RNS/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TranslateLib",
    ],
)

cc_library(
    name = "OpenFheUtils",
    srcs = ["OpenFheUtils.cpp"],
    hdrs = [
        "OpenFhePkeTemplates.h",
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
    ],
    deps = [
        ":OpenFheUtils",
        "@cereal",
        "@heir//lib/Analysis/SelectVariableNames",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Utils:TargetUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
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
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Utils:TargetUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "OpenFhePkePybindEmitter",
    srcs = ["OpenFhePkePybindEmitter.cpp"],
    hdrs = [
        "OpenFhePkePybindEmitter.h",
        "OpenFhePkeTemplates.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
