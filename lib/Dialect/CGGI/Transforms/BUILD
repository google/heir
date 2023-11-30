package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = [
        "@heir//include/Dialect/CGGI/Transforms:Passes.h",
    ],
    deps = [
        ":SetDefaultParameters",
        "@heir//include/Dialect/CGGI/Transforms:pass_inc_gen",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "SetDefaultParameters",
    srcs = ["SetDefaultParameters.cpp"],
    hdrs = [
        "@heir//include/Dialect/CGGI/Transforms:SetDefaultParameters.h",
    ],
    deps = [
        "@heir//include/Dialect/CGGI/Transforms:pass_inc_gen",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)
