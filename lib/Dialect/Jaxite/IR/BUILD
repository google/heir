package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "JaxiteDialect.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Jaxite/IR:JaxiteDialect.h",
        "@heir//include/Dialect/Jaxite/IR:JaxiteOps.h",
        "@heir//include/Dialect/Jaxite/IR:JaxiteTypes.h",
    ],
    deps = [
        "@heir//include/Dialect/Jaxite/IR:dialect_inc_gen",
        "@heir//include/Dialect/Jaxite/IR:ops_inc_gen",
        "@heir//include/Dialect/Jaxite/IR:types_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
