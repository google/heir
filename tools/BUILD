# HEIR tools

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Custom `mlir-opt` replacement that links our dialect and passes
cc_binary(
    name = "heir-opt",
    srcs = ["heir-opt.cpp"],
    includes = ["include"],
    deps = [
        "@heir//lib/Conversion/BGVToPoly",
        "@heir//lib/Conversion/MemrefToArith:ExpandCopy",
        "@heir//lib/Conversion/MemrefToArith:MemrefToArithRegistration",
        "@heir//lib/Conversion/PolyToStandard",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/EncryptedArith/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Poly/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/Secret/Transforms",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:TosaToLinalg",
        "@llvm-project//mlir:Transforms",
    ],
)

# Custom `mlir-translate` replacement that adds our custom translations
cc_binary(
    name = "heir-translate",
    srcs = ["heir-translate.cpp"],
    includes = ["include"],
    deps = [
        "@heir//lib/Target/Metadata:MetadataEmitter",
        "@heir//lib/Target/Verilog:VerilogEmitter",
        "@llvm-project//mlir:TranslateLib",
    ],
)
