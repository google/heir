load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "LowerUnpack",
    srcs = ["LowerUnpack.cpp"],
    hdrs = ["LowerUnpack.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Transforms/ConvertToCiphertextSemantics:AssignLayout",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "LowerUnpack",
    td_file = "LowerUnpack.td",
)
