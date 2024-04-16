package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = [
        "@heir//include/Dialect/BGV/Transforms:Passes.h",
    ],
    deps = [
        ":AddClientInterface",
        "@heir//include/Dialect/BGV/Transforms:pass_inc_gen",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "AddClientInterface",
    srcs = ["AddClientInterface.cpp"],
    hdrs = [
        "@heir//include/Dialect/BGV/Transforms:AddClientInterface.h",
    ],
    deps = [
        "@heir//include/Dialect/BGV/Transforms:pass_inc_gen",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
