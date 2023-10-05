package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "PolyToStandard",
    srcs = ["PolyToStandard.cpp"],
    hdrs = [
        "@heir//include/Conversion/PolyToStandard:PolyToStandard.h",
    ],
    deps = [
        "@heir//include/Conversion/PolyToStandard:pass_inc_gen",
        "@heir//lib/Conversion:Utils",
        "@heir//lib/Dialect/Poly/IR:PolyOps",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)
