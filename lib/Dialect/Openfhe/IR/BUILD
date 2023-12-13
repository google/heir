package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "OpenfheDialect.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Openfhe/IR:OpenfheDialect.h",
        "@heir//include/Dialect/Openfhe/IR:OpenfheOps.h",
        "@heir//include/Dialect/Openfhe/IR:OpenfheTypes.h",
    ],
    deps = [
        "@heir//include/Dialect/Openfhe/IR:dialect_inc_gen",
        "@heir//include/Dialect/Openfhe/IR:ops_inc_gen",
        "@heir//include/Dialect/Openfhe/IR:types_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
