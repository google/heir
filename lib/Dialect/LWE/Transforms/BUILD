package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = [
        "@heir//include/Dialect/LWE/Transforms:Passes.h",
    ],
    deps = [
        ":SetDefaultParameters",
        "@heir//include/Dialect/LWE/Transforms:pass_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "SetDefaultParameters",
    srcs = ["SetDefaultParameters.cpp"],
    hdrs = [
        "@heir//include/Dialect/LWE/Transforms:SetDefaultParameters.h",
    ],
    deps = [
        "@heir//include/Dialect/LWE/Transforms:pass_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)
