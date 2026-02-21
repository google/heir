load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = [
        "Passes.h",
    ],
    deps = [
        ":AllocToInPlace",
        ":ConfigureCryptoContext",
        ":ConvertToExtendedBasis",
        ":CountAddAndKeySwitch",
        ":FastRotationPrecompute",
        ":HoistKeySwitchDown",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
    ],
)

cc_library(
    name = "AllocToInPlace",
    srcs = ["AllocToInPlace.cpp"],
    hdrs = ["AllocToInPlace.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Utils:AllocToInPlaceUtils",
        "@llvm-project//llvm:Support",
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
    hdrs = [
        "ConfigureCryptoContext.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/MulDepthAnalysis",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Enums",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:TransformUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "CountAddAndKeySwitch",
    srcs = ["CountAddAndKeySwitch.cpp"],
    hdrs = [
        "CountAddAndKeySwitch.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/AddAndKeySwitchCountAnalysis",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect:ModuleAttributes",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "FastRotationPrecompute",
    srcs = ["FastRotationPrecompute.cpp"],
    hdrs = [
        "FastRotationPrecompute.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ConvertToExtendedBasis",
    srcs = ["ConvertToExtendedBasis.cpp"],
    hdrs = [
        "ConvertToExtendedBasis.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "HoistKeySwitchDown",
    srcs = ["HoistKeySwitchDown.cpp"],
    hdrs = [
        "HoistKeySwitchDown.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "Openfhe",
    td_file = "Passes.td",
)
