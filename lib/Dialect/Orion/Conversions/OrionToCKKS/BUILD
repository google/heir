load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OrionToCKKS",
    srcs = ["OrionToCKKS.cpp"],
    hdrs = ["OrionToCKKS.h"],
    deps = [
        ":IRMaterializingVisitor",
        ":pass_inc_gen",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Orion/IR:Dialect",
        "@heir//lib/Kernel:ArithmeticDag",
        "@heir//lib/Kernel:IRMaterializingVisitor",
        "@heir//lib/Kernel:KernelImplementation",
        "@heir//lib/Utils:ConversionUtils",
        "@heir//lib/Utils/Polynomial:ChebyshevPatersonStockmeyer",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "IRMaterializingVisitor",
    srcs = ["IRMaterializingVisitor.cpp"],
    hdrs = ["IRMaterializingVisitor.h"],
    deps = [
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Kernel:AbstractValue",
        "@heir//lib/Kernel:ArithmeticDag",
        "@heir//lib/Kernel:KernelImplementation",
        "@heir//lib/Utils",
        "@heir//lib/Utils:MathUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

add_heir_transforms(
    header_filename = "OrionToCKKS.h.inc",
    pass_name = "OrionToCKKS",
    td_file = "OrionToCKKS.td",
)
