# Polynomial dialect

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "PolynomialDialect.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Polynomial/IR:PolynomialAttributes.h",
        "@heir//include/Dialect/Polynomial/IR:PolynomialDialect.h",
        "@heir//include/Dialect/Polynomial/IR:PolynomialOps.h",
        "@heir//include/Dialect/Polynomial/IR:PolynomialTypes.h",
    ],
    deps = [
        ":Polynomial",
        ":PolynomialAttributes",
        ":PolynomialOps",
        ":PolynomialTypes",
        "@heir//include/Dialect/Polynomial/IR:attributes_inc_gen",
        "@heir//include/Dialect/Polynomial/IR:canonicalize_inc_gen",
        "@heir//include/Dialect/Polynomial/IR:dialect_inc_gen",
        "@heir//include/Dialect/Polynomial/IR:ops_inc_gen",
        "@heir//include/Dialect/Polynomial/IR:types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)

cc_library(
    name = "PolynomialAttributes",
    srcs = [
        "PolynomialAttributes.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Polynomial/IR:PolynomialAttributes.h",
        "@heir//include/Dialect/Polynomial/IR:PolynomialDialect.h",
        "@heir//include/Dialect/Polynomial/IR:StaticRoots.h",
    ],
    deps = [
        ":NumberTheory",
        ":Polynomial",
        "@heir//include/Dialect/Polynomial/IR:attributes_inc_gen",
        "@heir//include/Dialect/Polynomial/IR:dialect_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "PolynomialOps",
    srcs = [
        "PolynomialOps.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Polynomial/IR:PolynomialDialect.h",
        "@heir//include/Dialect/Polynomial/IR:PolynomialOps.h",
        "@heir//include/Dialect/Polynomial/IR:PolynomialTypes.h",
        "@heir//include/Dialect/RNS/IR:RNSTypeInterfaces.h",
    ],
    deps = [
        ":PolynomialAttributes",
        "@heir//include/Dialect/Polynomial/IR:canonicalize_inc_gen",
        "@heir//include/Dialect/Polynomial/IR:dialect_inc_gen",
        "@heir//include/Dialect/Polynomial/IR:ops_inc_gen",
        "@heir//include/Dialect/Polynomial/IR:types_inc_gen",
        "@heir//include/Dialect/RNS/IR:type_interfaces_inc_gen",
        "@heir//lib/Dialect/RNS/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "PolynomialTypes",
    srcs = [
        "PolynomialTypes.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Polynomial/IR:PolynomialTypes.h",
        "@heir//include/Dialect/RNS/IR:RNSTypeInterfaces.h",
    ],
    deps = [
        ":PolynomialAttributes",
        "@heir//include/Dialect/Polynomial/IR:types_inc_gen",
        "@heir//include/Dialect/RNS/IR:type_interfaces_inc_gen",
        "@heir//lib/Dialect/RNS/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "Polynomial",
    srcs = [
        "Polynomial.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Polynomial/IR:Polynomial.h",
        "@heir//include/Dialect/Polynomial/IR:PolynomialDetail.h",
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

cc_library(
    name = "NumberTheory",
    srcs = [
        "NumberTheory.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Polynomial/IR:NumberTheory.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "NumberTheoryTest",
    size = "small",
    srcs = ["NumberTheoryTest.cpp"],
    deps = [
        ":NumberTheory",
        "@googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)
