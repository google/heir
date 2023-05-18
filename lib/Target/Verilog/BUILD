# Verilog Emitter

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = [
        "@heir//:internal",
    ],
)

cc_library(
    name = "VerilogEmitter",
    srcs = ["VerilogEmitter.cpp"],
    hdrs = ["@heir//include/Target/Verilog:VerilogEmitter.h"],
    deps = [
        "@heir//include/Target/Verilog:verilog_emitter",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)
