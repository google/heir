package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "TfheRustBoolDialect.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/TfheRustBool/IR:TfheRustBoolDialect.h",
        "@heir//include/Dialect/TfheRustBool/IR:TfheRustBoolOps.h",
        "@heir//include/Dialect/TfheRustBool/IR:TfheRustBoolTypes.h",
    ],
    deps = [
        "@heir//include/Dialect/TfheRustBool/IR:dialect_inc_gen",
        "@heir//include/Dialect/TfheRustBool/IR:ops_inc_gen",
        "@heir//include/Dialect/TfheRustBool/IR:types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
