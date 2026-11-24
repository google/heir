load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CheddarGPUToGPU",
    srcs = [
        "CheddarGPUToGPU.cpp",
        "KernelBuilders.cpp",
    ],
    hdrs = [
        "CheddarGPUToGPU.h",
        "KernelBuilders.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/CheddarGPU/IR:Dialect",
        "@heir//lib/Dialect/CheddarGPU/Utils:ConstantTables",
        "@heir//lib/Utils:AttributeUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    header_filename = "CheddarGPUToGPU.h.inc",
    pass_name = "CheddarGPUToGPU",
    td_file = "Passes.td",
)
