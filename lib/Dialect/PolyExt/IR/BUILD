# PolyExt dialect

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "PolyExtDialect.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/PolyExt/IR:PolyExtDialect.h",
        "@heir//include/Dialect/PolyExt/IR:PolyExtOps.h",
    ],
    deps = [
        "@heir//include/Dialect/PolyExt/IR:dialect_inc_gen",
        "@heir//include/Dialect/PolyExt/IR:ops_inc_gen",
        "@heir//lib/Dialect/Poly/IR:Dialect",
        "@heir//lib/Dialect/Poly/IR:Polynomial",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
