load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = ["Passes.h"],
    deps = [
        ":AllocToInPlace",
        ":ConfigureCryptoContext",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Lattigo/IR:Dialect",
    ],
)

cc_library(
    name = "AllocToInPlace",
    srcs = ["AllocToInPlace.cpp"],
    hdrs = ["AllocToInPlace.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Lattigo/IR:Dialect",
        "@heir//lib/Utils:AllocToInPlaceUtils",
        "@heir//lib/Utils/Tablegen:InPlaceOpInterface",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
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
        ":pass_inc_gen",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/Lattigo/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:TransformUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "Lattigo",
    td_file = "Passes.td",
)
