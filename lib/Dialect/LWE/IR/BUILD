package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "LWEDialect.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/LWE/IR:LWEAttributes.h",
        "@heir//include/Dialect/LWE/IR:LWEDialect.h",
        "@heir//include/Dialect/LWE/IR:LWEOps.h",
        "@heir//include/Dialect/LWE/IR:LWETypes.h",
    ],
    deps = [
        "@heir//include/Dialect/LWE/IR:attributes_inc_gen",
        "@heir//include/Dialect/LWE/IR:dialect_inc_gen",
        "@heir//include/Dialect/LWE/IR:ops_inc_gen",
        "@heir//include/Dialect/LWE/IR:types_inc_gen",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Polynomial",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)
