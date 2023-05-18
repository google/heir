# Verilog Emitter
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = [
        "@heir//:internal",
    ],
)

exports_files(
    srcs = ["VerilogEmitter.h"],
    visibility = ["@heir//:internal"],
)

cc_library(
    name = "verilog_emitter",
    hdrs = ["VerilogEmitter.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
