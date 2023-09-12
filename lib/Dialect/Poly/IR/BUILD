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
        "@heir//include/Dialect/Poly/IR:PolyAttributes.h",
        "@heir//include/Dialect/Poly/IR:PolyDialect.h",
        "@heir//include/Dialect/Poly/IR:PolyOps.h",
        "@heir//include/Dialect/Poly/IR:PolyTypes.h",
    ],
    includes = ["@heir//include"],
    deps = [
        ":PolyAttributes",
        ":PolyOps",
        ":Polynomial",
        "@heir//include/Dialect/Poly/IR:attributes_inc_gen",
        "@heir//include/Dialect/Poly/IR:dialect_inc_gen",
        "@heir//include/Dialect/Poly/IR:ops_inc_gen",
        "@heir//include/Dialect/Poly/IR:types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)

cc_library(
    name = "PolyAttributes",
    srcs = [
        "PolyAttributes.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Poly/IR:PolyAttributes.h",
        "@heir//include/Dialect/Poly/IR:PolyDialect.h",
    ],
    deps = [
        ":Polynomial",
        "@heir//include/Dialect/Poly/IR:attributes_inc_gen",
        "@heir//include/Dialect/Poly/IR:dialect_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "PolyOps",
    srcs = [
        "PolyOps.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Poly/IR:PolyDialect.h",
        "@heir//include/Dialect/Poly/IR:PolyOps.h",
        "@heir//include/Dialect/Poly/IR:PolyTypes.h",
    ],
    includes = ["@heir//include"],
    deps = [
        ":PolyAttributes",
        "@heir//include/Dialect/Poly/IR:dialect_inc_gen",
        "@heir//include/Dialect/Poly/IR:ops_inc_gen",
        "@heir//include/Dialect/Poly/IR:types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "Polynomial",
    srcs = [
        "Polynomial.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Poly/IR:Polynomial.h",
        "@heir//include/Dialect/Poly/IR:PolynomialDetail.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "PolynomialTest",
    size = "small",
    srcs = ["PolynomialTest.cpp"],
    deps = [
        ":Polynomial",
        "@googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
