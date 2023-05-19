# HEIR tools

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = [
        "@heir//:internal",
    ],
)

# Custom `mlir-opt` replacement that links our dialect and passes
cc_binary(
    name = "heir-opt",
    srcs = ["heir-opt.cpp"],
    includes = ["include"],
    deps = [
        "@heir//lib/Conversion/MemrefToArith:MemrefToArithRegistration",
        "@heir//lib/Dialect/EncryptedArith/IR:Dialect",
        "@heir//lib/Dialect/HEIR/IR:Dialect",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:SCFDialect",
    ],
)

# Custom `mlir-translate` replacement that adds our custom translations
cc_binary(
    name = "heir-translate",
    srcs = ["heir-translate.cpp"],
    includes = ["include"],
    deps = [
        "@heir//lib/Target/Verilog:VerilogEmitter",
        "@llvm-project//mlir:TranslateLib",
    ],
)
