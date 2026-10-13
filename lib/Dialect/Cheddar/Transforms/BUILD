load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CheddarBufferize",
    srcs = ["CheddarBufferize.cpp"],
    hdrs = ["CheddarBufferize.h"],
    deps = [
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "BufferizableOpInterfaceImpl",
    srcs = ["BufferizableOpInterfaceImpl.cpp"],
    hdrs = ["BufferizableOpInterfaceImpl.h"],
    deps = [
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@heir//lib/Dialect/Cheddar/IR:ops_inc_gen",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationInterfaces",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "FuseOps",
    srcs = ["FuseOps.cpp"],
    hdrs = ["FuseOps.h"],
    deps = [
        ":fuse_ops_inc_gen",
        ":fuse_ops_rewriters_inc_gen",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    generated_target_name = "fuse_ops_inc_gen",
    header_filename = "FuseOps.h.inc",
    pass_name = "CheddarFuseOps",
    td_file = "FuseOps.td",
)

cc_library(
    name = "ConfigureCryptoContext",
    srcs = ["ConfigureCryptoContext.cpp"],
    hdrs = ["ConfigureCryptoContext.h"],
    deps = [
        ":configure_crypto_context_inc_gen",
        "@heir//lib/Analysis/RotationAnalysis",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:TransformUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

add_heir_transforms(
    generated_target_name = "configure_crypto_context_inc_gen",
    header_filename = "ConfigureCryptoContext.h.inc",
    pass_name = "ConfigureCryptoContext",
    td_file = "ConfigureCryptoContext.td",
)

gentbl_cc_library(
    name = "fuse_ops_rewriters_inc_gen",
    tbl_outs = {"FuseOpsPatterns.cpp.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FuseOpsPatterns.td",
    deps = [
        "@heir//lib/Dialect/Cheddar/IR:ops_inc_gen",
        "@heir//lib/Dialect/Cheddar/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)
