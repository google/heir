load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ConvertToCiphertextSemantics",
    srcs = ["ConvertToCiphertextSemantics.cpp"],
    hdrs = ["ConvertToCiphertextSemantics.h"],
    deps = [
        ":AssignLayout",
        ":TypeConversion",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Secret/IR:SecretAttributes",
        "@heir//lib/Dialect/Secret/IR:SecretPatterns",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Kernel",
        "@heir//lib/Kernel:AbstractValue",
        "@heir//lib/Kernel:ArithmeticDag",
        "@heir//lib/Kernel:IRMaterializingVisitor",
        "@heir//lib/Kernel:KernelImplementation",
        "@heir//lib/Transforms/DropUnitDims",
        "@heir//lib/Utils",
        "@heir//lib/Utils:AttributeUtils",
        "@heir//lib/Utils:ContextAwareConversionUtils",
        "@heir//lib/Utils:ContextAwareDialectConversion",
        "@heir//lib/Utils:ContextAwareTypeConversion",
        "@heir//lib/Utils:MathUtils",
        "@heir//lib/Utils/Layout:Codegen",
        "@heir//lib/Utils/Layout:Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "TypeConversion",
    srcs = ["TypeConversion.cpp"],
    hdrs = ["TypeConversion.h"],
    deps = [
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "AssignLayout",
    srcs = ["AssignLayout.cpp"],
    hdrs = ["AssignLayout.h"],
    deps = [
        ":TypeConversion",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Utils/Layout:Codegen",
        "@heir//lib/Utils/Layout:Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "ConvertToCiphertextSemantics",
    td_file = "ConvertToCiphertextSemantics.td",
)
