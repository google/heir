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
        "@heir//include/Dialect/BGV/IR:BGVAttributes.h",
        "@heir//include/Dialect/BGV/IR:BGVDialect.h",
        "@heir//include/Dialect/BGV/IR:BGVOps.h",
        "@heir//include/Dialect/BGV/IR:BGVTraits.h",
        "@heir//include/Dialect/BGV/IR:BGVTypes.h",
    ],
    includes = ["@heir//include"],
    deps = [
        "@heir//include/Dialect/BGV/IR:attributes_inc_gen",
        "@heir//include/Dialect/BGV/IR:dialect_inc_gen",
        "@heir//include/Dialect/BGV/IR:ops_inc_gen",
        "@heir//include/Dialect/BGV/IR:types_inc_gen",
        "@heir//include/Dialect/Poly/IR:attributes_inc_gen",
        "@heir//lib/Dialect/Poly/IR:PolyAttributes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
