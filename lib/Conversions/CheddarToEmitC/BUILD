load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CheddarToEmitC",
    srcs = ["CheddarToEmitC.cpp"],
    hdrs = ["CheddarToEmitC.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToEmitC",
        "@llvm-project//mlir:EmitCDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToEmitC",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToEmitC",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    header_filename = "CheddarToEmitC.h.inc",
    pass_name = "CheddarToEmitC",
    td_file = "CheddarToEmitC.td",
)
