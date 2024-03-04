# TensorExt dialect implementation

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "TensorExtDialect.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/TensorExt/IR:TensorExtDialect.h",
        "@heir//include/Dialect/TensorExt/IR:TensorExtOps.h",
    ],
    deps = [
        ":TensorExtOps",
        "@heir//include/Dialect/TensorExt/IR:dialect_inc_gen",
        "@heir//include/Dialect/TensorExt/IR:ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "TensorExtOps",
    srcs = [
        "TensorExtOps.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/TensorExt/IR:TensorExtDialect.h",
        "@heir//include/Dialect/TensorExt/IR:TensorExtOps.h",
    ],
    deps = [
        "@heir//include/Dialect/TensorExt/IR:dialect_inc_gen",
        "@heir//include/Dialect/TensorExt/IR:ops_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
