package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "SecretToBGV",
    srcs = ["SecretToBGV.cpp"],
    hdrs = [
        "@heir//include/Conversion/SecretToBGV:SecretToBGV.h",
    ],
    deps = [
        "@heir//include/Conversion/SecretToBGV:pass_inc_gen",
        "@heir//lib/Conversion:Utils",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Polynomial",
        "@heir//lib/Dialect/Polynomial/IR:PolynomialAttributes",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
