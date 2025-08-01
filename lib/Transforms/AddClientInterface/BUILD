load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "AddClientInterface",
    srcs = ["AddClientInterface.cpp"],
    hdrs = ["AddClientInterface.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Transforms/ConvertToCiphertextSemantics:AssignLayout",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "AddClientInterface",
    td_file = "AddClientInterface.td",
)
