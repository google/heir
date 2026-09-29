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
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ConvertToEmitCInterface",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:EmitCDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToEmitC",
        "@llvm-project//mlir:MemRefUtils",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    header_filename = "CheddarToEmitC.h.inc",
    pass_name = "CheddarEmitCBoundary",
    td_file = "CheddarToEmitC.td",
)
