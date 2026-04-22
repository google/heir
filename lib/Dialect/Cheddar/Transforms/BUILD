load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "FuseOps",
    srcs = ["FuseOps.cpp"],
    hdrs = ["FuseOps.h"],
    deps = [
        ":fuse_ops_inc_gen",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "ConfigureCryptoContext",
    srcs = ["ConfigureCryptoContext.cpp"],
    hdrs = ["ConfigureCryptoContext.h"],
    deps = [
        ":configure_crypto_context_inc_gen",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

add_heir_transforms(
    generated_target_name = "fuse_ops_inc_gen",
    header_filename = "FuseOps.h.inc",
    pass_name = "CheddarFuseOps",
    td_file = "FuseOps.td",
)

add_heir_transforms(
    generated_target_name = "configure_crypto_context_inc_gen",
    header_filename = "ConfigureCryptoContext.h.inc",
    pass_name = "ConfigureCryptoContext",
    td_file = "ConfigureCryptoContext.td",
)
