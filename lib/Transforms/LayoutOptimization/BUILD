load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "LayoutOptimization",
    srcs = ["LayoutOptimization.cpp"],
    hdrs = ["LayoutOptimization.h"],
    deps = [
        ":Hoisting",
        ":InterfaceImpl",
        ":LayoutConversionCost",
        ":Patterns",
        ":pass_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Kernel",
        "@heir//lib/Utils:AttributeUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgInterfaces",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "Hoisting",
    srcs = ["Hoisting.h"],
    hdrs = [],
    deps = [
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Kernel",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "InterfaceImpl",
    srcs = ["InterfaceImpl.cpp"],
    hdrs = [
        "Hoisting.h",
        "InterfaceImpl.h",
    ],
    deps = [
        ":Hoisting",
        "@heir//lib/Dialect/TensorExt/IR:TensorExtAttributes",
        "@heir//lib/Dialect/TensorExt/IR:TensorExtOps",
        "@heir//lib/Kernel",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "Patterns",
    srcs = ["Patterns.cpp"],
    hdrs = ["Patterns.h"],
    deps = [
        "@heir//lib/Dialect/Secret/IR:SecretPatterns",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Utils:AttributeUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "LayoutConversionCost",
    srcs = ["LayoutConversionCost.cpp"],
    hdrs = ["LayoutConversionCost.h"],
    deps = [
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/Transforms:ImplementShiftNetwork",
        "@heir//lib/Utils",
        "@heir//lib/Utils:AttributeUtils",
        "@heir//lib/Utils:TransformUtils",
        "@llvm-project//llvm:Support",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "LayoutOptimization",
    td_file = "LayoutOptimization.td",
)
