# TfheRustBool Emitter
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    srcs = ["TfheRustBoolEmitter.h"],
)

cc_library(
    name = "tfherustbool_emitter",
    hdrs = [
        "TfheRustBoolEmitter.h",
    ],
    deps = [
        "@heir//lib/Analysis/SelectVariableNames",
        "@heir//lib/Dialect/TfheRustBool/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)
