package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "BGVToPoly",
    srcs = ["BGVToPoly.cpp"],
    hdrs = [
        "@heir//include/Conversion/BGVToPoly:BGVToPoly.h",
    ],
    deps = [
        "@heir//include/Conversion/BGVToPoly:pass_inc_gen",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/Poly/IR:Dialect",
        "@heir//lib/Dialect/Poly/IR:Polynomial",
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
