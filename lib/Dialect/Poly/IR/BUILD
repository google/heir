# Poly dialect

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "PolyDialect.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Poly/IR:PolyDialect.h",
        "@heir//include/Dialect/Poly/IR:PolyOps.h",
        "@heir//include/Dialect/Poly/IR:PolyTypes.h",
    ],
    includes = ["@heir//include"],
    deps = [
        "@heir//include/Dialect/Poly/IR:dialect_inc_gen",
        "@heir//include/Dialect/Poly/IR:ops_inc_gen",
        "@heir//include/Dialect/Poly/IR:types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
