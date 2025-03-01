load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ConvertToCiphertextSemantics",
    srcs = ["ConvertToCiphertextSemantics.cpp"],
    hdrs = ["ConvertToCiphertextSemantics.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:ContextAwareDialectConversion",
        "@heir//lib/Utils:ContextAwareTypeConversion",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "ConvertToCiphertextSemantics",
    td_file = "ConvertToCiphertextSemantics.td",
)
