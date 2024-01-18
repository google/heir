package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CombToCGGI",
    srcs = ["CombToCGGI.cpp"],
    hdrs = [
        "@heir//include/Conversion/CombToCGGI:CombToCGGI.h",
    ],
    deps = [
        "@heir//include/Conversion/CombToCGGI:pass_inc_gen",
        "@heir//lib/Conversion:Utils",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/Comb/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)
