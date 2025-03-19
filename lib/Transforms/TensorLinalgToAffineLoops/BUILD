load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "TensorLinalgToAffineLoops",
    srcs = ["TensorLinalgToAffineLoops.cpp"],
    hdrs = ["TensorLinalgToAffineLoops.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Utils/RewriteUtils",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgUtils",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "TensorLinalgToAffineLoops",
    td_file = "TensorLinalgToAffineLoops.td",
)
