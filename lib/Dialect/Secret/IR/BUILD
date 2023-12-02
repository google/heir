package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "SecretDialect.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Secret/IR:SecretDialect.h",
        "@heir//include/Dialect/Secret/IR:SecretOps.h",
        "@heir//include/Dialect/Secret/IR:SecretTypes.h",
    ],
    deps = [
        ":SecretOps",
        ":SecretPatterns",
        "@heir//include/Dialect/Secret/IR:dialect_inc_gen",
        "@heir//include/Dialect/Secret/IR:ops_inc_gen",
        "@heir//include/Dialect/Secret/IR:types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)

cc_library(
    name = "SecretPatterns",
    srcs = [
        "SecretPatterns.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Secret/IR:SecretDialect.h",
        "@heir//include/Dialect/Secret/IR:SecretOps.h",
        "@heir//include/Dialect/Secret/IR:SecretPatterns.h",
        "@heir//include/Dialect/Secret/IR:SecretTypes.h",
    ],
    deps = [
        "@heir//include/Dialect/Secret/IR:dialect_inc_gen",
        "@heir//include/Dialect/Secret/IR:ops_inc_gen",
        "@heir//include/Dialect/Secret/IR:types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "SecretOps",
    srcs = [
        "SecretOps.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Secret/IR:SecretDialect.h",
        "@heir//include/Dialect/Secret/IR:SecretOps.h",
        "@heir//include/Dialect/Secret/IR:SecretTypes.h",
    ],
    deps = [
        ":SecretPatterns",
        "@heir//include/Dialect/Secret/IR:dialect_inc_gen",
        "@heir//include/Dialect/Secret/IR:ops_inc_gen",
        "@heir//include/Dialect/Secret/IR:types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)
