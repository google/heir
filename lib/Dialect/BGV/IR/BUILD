# BGV dialect

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "BGVDialect.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/BGV/IR:BGVDialect.h",
        "@heir//include/Dialect/BGV/IR:BGVOps.h",
    ],
    deps = [
        "@heir//include/Dialect/BGV/IR:dialect_inc_gen",
        "@heir//include/Dialect/BGV/IR:ops_inc_gen",
        "@heir//include/Dialect/Polynomial/IR:attributes_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:PolynomialAttributes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)
