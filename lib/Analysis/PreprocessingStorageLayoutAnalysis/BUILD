load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "PreprocessingStorageLayoutAnalysis",
    srcs = ["PreprocessingStorageLayoutAnalysis.cpp"],
    hdrs = ["PreprocessingStorageLayoutAnalysis.h"],
    deps = [
        "@heir//lib/Dialect/Preprocessing/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
    ],
)
