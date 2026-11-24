load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "BufferizableOpInterfaceImpl",
    srcs = ["BufferizableOpInterfaceImpl.cpp"],
    hdrs = ["BufferizableOpInterfaceImpl.h"],
    deps = [
        "@heir//lib/Dialect/CheddarGPU/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationInterfaces",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
