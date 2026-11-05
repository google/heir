# HEIR tools
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# https://bazel.build/remote/output-directories#layout-diagram
WORKSPACE_PATH = ""

# Custom `mlir-opt` replacement that links our dialect and passes
cc_binary(
    name = "heir-opt",
    srcs = ["heir-opt.cpp"],
    data = [
        "@abc//:abc_bin",
        "@heir//lib/Transforms/YosysOptimizer/yosys:techmap_lut3.v",
        "@heir//lib/Transforms/YosysOptimizer/yosys:techmap_lut4.v",
    ],
    # The strings in these defines are passed through various pre-processors,
    # including the materializing of quoted strings as done in bash, so to
    # ensure that the #define'd variables stay quoted strings in the generated
    # C++, we need to enclose them in `\\\"` (an escaped backslash followed by
    # an escaped quote).
    defines = [
        "HEIR_ABC_BINARY=\\\"$(rootpath @abc//:abc_bin)\\\"",
        "HEIR_YOSYS_SCRIPTS_DIR=\\\"" + WORKSPACE_PATH + "lib/Transforms/YosysOptimizer/yosys\\\"",
    ],
    includes = ["include"],
    deps = [
        "@heir//lib/Tools:HeirOptRegistration",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
    ],
)

# Custom `mlir-translate` replacement that adds our custom translations
cc_binary(
    name = "heir-translate",
    srcs = ["heir-translate.cpp"],
    includes = ["include"],
    deps = [
        "@heir//lib/Tools:HeirTranslateRegistration",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)

cc_binary(
    name = "heir-lsp",
    srcs = ["heir-lsp.cpp"],
    includes = ["include"],
    deps = [
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@heir//lib/Dialect/Comb/IR:Dialect",
        "@heir//lib/Dialect/Debug/IR:Dialect",
        "@heir//lib/Dialect/Jaxite/IR:Dialect",
        "@heir//lib/Dialect/JaxiteWord/IR:Dialect",
        "@heir//lib/Dialect/Kernel/IR:Dialect",
        "@heir//lib/Dialect/KeyMgmt/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Lattigo/IR:Dialect",
        "@heir//lib/Dialect/MathExt/IR:Dialect",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Dialect/Preprocessing/IR:Dialect",
        "@heir//lib/Dialect/RNS/IR:Dialect",
        "@heir//lib/Dialect/Random/IR:Dialect",
        "@heir//lib/Dialect/Rotom/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Dialect/TfheRust/IR:Dialect",
        "@heir//lib/Dialect/TfheRustBool/IR:Dialect",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MlirLspServerLib",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

bzl_library(
    name = "heir_translate_bzl",
    srcs = ["heir-translate.bzl"],
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "heir_opt_bzl",
    srcs = ["heir-opt.bzl"],
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "heir_jaxite_bzl",
    srcs = ["heir-jaxite.bzl"],
    visibility = ["//visibility:public"],
    deps = [
        ":heir_opt_bzl",
        ":heir_translate_bzl",
        "@rules_python//python:py_library_bzl",
    ],
)

bzl_library(
    name = "heir_openfhe_bzl",
    srcs = ["heir-openfhe.bzl"],
    visibility = ["//visibility:public"],
    deps = [
        ":heir_opt_bzl",
        ":heir_translate_bzl",
    ],
)

bzl_library(
    name = "heir_cheddar_bzl",
    srcs = ["heir-cheddar.bzl"],
    visibility = ["//visibility:public"],
    deps = [
        ":heir_opt_bzl",
        ":heir_translate_bzl",
    ],
)

bzl_library(
    name = "heir_tfhe_rs",
    srcs = ["heir-tfhe-rs.bzl"],
    visibility = ["//visibility:public"],
    deps = [
        ":heir_opt_bzl",
        ":heir_translate_bzl",
    ],
)

bzl_library(
    name = "llc_bzl",
    srcs = ["llc.bzl"],
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "mlir_translate_bzl",
    srcs = ["mlir-translate.bzl"],
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "mlir_opt_gpu_bzl",
    srcs = ["mlir-opt-gpu.bzl"],
    visibility = ["//visibility:public"],
)
