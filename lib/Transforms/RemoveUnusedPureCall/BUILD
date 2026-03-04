load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "RemoveUnusedPureCall",
    srcs = ["RemoveUnusedPureCall.cpp"],
    hdrs = [
        "RemoveUnusedPureCall.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect:ModuleAttributes",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "RemoveUnusedPureCall",
)
