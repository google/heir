# Verilog Emitter

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "VerilogEmitter",
    srcs = ["VerilogEmitter.cpp"],
    hdrs = ["@heir//include/Target/Verilog:VerilogEmitter.h"],
    deps = [
        "@heir//include/Target/Verilog:verilog_emitter",
        "@heir//lib/Conversion/MemrefToArith:Utils",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Target:Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)
