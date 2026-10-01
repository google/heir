load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CheddarBufferize",
    srcs = ["CheddarBufferize.cpp"],
    hdrs = ["CheddarBufferize.h"],
    deps = [
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "BufferizableOpInterfaceImpl",
    srcs = ["BufferizableOpInterfaceImpl.cpp"],
    hdrs = ["BufferizableOpInterfaceImpl.h"],
    deps = [
        "@heir//lib/Dialect/Cheddar/IR:Dialect",
        "@heir//lib/Dialect/Cheddar/IR:ops_inc_gen",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationInterfaces",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
