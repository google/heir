# PISA Emitter

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "PISAEmitter",
    srcs = ["PISAEmitter.cpp"],
    hdrs = [
        "PISAEmitter.h",
    ],
    deps = [
        "@heir//lib/Analysis/SelectVariableNames",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/PISA/IR:Dialect",
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
