package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "BGVToPolynomial",
    srcs = ["BGVToPolynomial.cpp"],
    hdrs = [
        "@heir//include/Conversion/BGVToPolynomial:BGVToPolynomial.h",
    ],
    deps = [
        "@heir//include/Conversion/BGVToPolynomial:pass_inc_gen",
        "@heir//lib/Conversion:Utils",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Polynomial",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)
