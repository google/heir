# Lattigo Emitter

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "LattigoEmitter",
    srcs = ["LattigoEmitter.cpp"],
    hdrs = [
        "LattigoEmitter.h",
        "LattigoTemplates.h",
    ],
    deps = [
        "@heir//lib/Analysis/SelectVariableNames",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/Lattigo/IR:Dialect",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/RNS/IR:Dialect",
        "@heir//lib/Utils:TargetUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TranslateLib",
    ],
)
