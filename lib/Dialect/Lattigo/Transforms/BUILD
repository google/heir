load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = ["Passes.h"],
    deps = [
        ":ConfigureCryptoContext",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Lattigo/IR:Dialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "ConfigureCryptoContext",
    srcs = ["ConfigureCryptoContext.cpp"],
    hdrs = ["ConfigureCryptoContext.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Lattigo/IR:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "Lattigo",
    td_file = "Passes.td",
)
