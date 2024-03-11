package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = [
        "@heir//include/Dialect/TensorExt/Transforms:Passes.h",
    ],
    deps = [
        ":CollapseInsertionChains",
        ":InsertRotate",
        "@heir//include/Dialect/TensorExt/Transforms:pass_inc_gen",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "InsertRotate",
    srcs = ["InsertRotate.cpp"],
    hdrs = [
        "@heir//include/Dialect/TensorExt/Transforms:InsertRotate.h",
    ],
    deps = [
        "@heir//include/Dialect/TensorExt/IR:canonicalize_inc_gen",
        "@heir//include/Dialect/TensorExt/Transforms:insert_rotate_inc_gen",
        "@heir//include/Dialect/TensorExt/Transforms:pass_inc_gen",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "CollapseInsertionChains",
    srcs = ["CollapseInsertionChains.cpp"],
    hdrs = [
        "@heir//include/Dialect/TensorExt/Transforms:CollapseInsertionChains.h",
    ],
    deps = [
        "@heir//include/Dialect/TensorExt/Transforms:pass_inc_gen",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)
