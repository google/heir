# TfheRust Emitter

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "TfheRustEmitter",
    srcs = ["TfheRustEmitter.cpp"],
    hdrs = [
        "TfheRustTemplates.h",
        "@heir//include/Target/TfheRust:TfheRustEmitter.h",
    ],
    deps = [
        "@heir//lib/Analysis/SelectVariableNames",
        "@heir//lib/Dialect/TfheRust/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)
