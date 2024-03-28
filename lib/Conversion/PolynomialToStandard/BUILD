package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "PolynomialToStandard",
    srcs = ["PolynomialToStandard.cpp"],
    hdrs = [
        "@heir//include/Conversion/PolynomialToStandard:PolynomialToStandard.h",
    ],
    deps = [
        "@heir//include/Conversion/PolynomialToStandard:pass_inc_gen",
        "@heir//lib/Conversion:Utils",
        "@heir//lib/Dialect/Polynomial/IR:Polynomial",
        "@heir//lib/Dialect/Polynomial/IR:PolynomialAttributes",
        "@heir//lib/Dialect/Polynomial/IR:PolynomialOps",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)
