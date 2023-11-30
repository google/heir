# HEIR tools

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

WORKSPACE_PATH = ""

# Custom `mlir-opt` replacement that links our dialect and passes
cc_binary(
    name = "heir-opt",
    srcs = ["heir-opt.cpp"],
    data = select({
        "@heir//:disable_yosys": [],
        "//conditions:default": [
            "@edu_berkeley_abc//:abc",
            "@heir//lib/Transforms/YosysOptimizer/yosys:share_files",
        ],
    }),
    defines = select({
        "@heir//:disable_yosys": ["HEIR_NO_YOSYS=1"],
        "//conditions:default": [],
    }),
    # Using the location directive to find share files results in multiple expanded paths.
    env = select({
        "@heir//:disable_yosys": {
        },
        "//conditions:default": {
            "HEIR_ABC_BINARY": "$(location @edu_berkeley_abc//:abc)",
            "HEIR_YOSYS_SCRIPTS_DIR": WORKSPACE_PATH + "lib/Transforms/YosysOptimizer/yosys",
        },
    }),
    includes = ["include"],
    deps = [
        "@heir//lib/Conversion/BGVToPolynomial",
        "@heir//lib/Conversion/CGGIToTfheRust",
        "@heir//lib/Conversion/CombToCGGI",
        "@heir//lib/Conversion/MemrefToArith:ExpandCopy",
        "@heir//lib/Conversion/MemrefToArith:MemrefToArithRegistration",
        "@heir//lib/Conversion/PolynomialToStandard",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/Comb/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/PolyExt/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/Secret/Transforms",
        "@heir//lib/Dialect/TfheRust/IR:Dialect",
        "@heir//lib/Transforms/Secretize",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IndexToLLVM",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:TensorToLinalg",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:TosaToArith",
        "@llvm-project//mlir:TosaToLinalg",
        "@llvm-project//mlir:TosaToTensor",
        "@llvm-project//mlir:Transforms",
    ] + select({
        "@heir//:disable_yosys": [],
        "//conditions:default": [
            "@heir//lib/Transforms/YosysOptimizer",
        ],
    }),
)

# Custom `mlir-translate` replacement that adds our custom translations
cc_binary(
    name = "heir-translate",
    srcs = ["heir-translate.cpp"],
    includes = ["include"],
    deps = [
        "@heir//lib/Target/Metadata:MetadataEmitter",
        "@heir//lib/Target/TfheRust:TfheRustEmitter",
        "@heir//lib/Target/Verilog:VerilogEmitter",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)
