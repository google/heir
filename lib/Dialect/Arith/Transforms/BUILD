load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = ["Passes.h"],
    deps = [
        ":QuarterWideInt",
        ":pass_inc_gen",
    ],
)

cc_library(
    name = "QuarterWideInt",
    srcs = ["QuarterWideInt.cpp"],
    hdrs = ["QuarterWideInt.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "Arith",
    td_file = "Passes.td",
)
