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
    ],
    includes = ["@heir//include"],
    deps = [
        "@heir//include/Dialect/LWE/IR:attributes_inc_gen",
        "@heir//include/Dialect/LWE/IR:dialect_inc_gen",
        "@heir//lib/Dialect/Poly/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
