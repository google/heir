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
    ],
    includes = ["@heir//include"],
    deps = [
        ":SecretOps",
        "@heir//include/Dialect/Secret/IR:dialect_inc_gen",
        "@heir//include/Dialect/Secret/IR:ops_inc_gen",
        "@heir//include/Dialect/Secret/IR:types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
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
    includes = ["@heir//include"],
    deps = [
        "@heir//include/Dialect/Secret/IR:dialect_inc_gen",
        "@heir//include/Dialect/Secret/IR:ops_inc_gen",
        "@heir//include/Dialect/Secret/IR:types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
