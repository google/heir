# TfheRustBool Emitter

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
        "@heir//lib/Target:Utils",
        "@heir//lib/Target/TfheRust:Utils",
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
