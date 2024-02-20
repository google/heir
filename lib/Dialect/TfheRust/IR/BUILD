package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "TfheRustDialect.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/TfheRust/IR:TfheRustDialect.h",
        "@heir//include/Dialect/TfheRust/IR:TfheRustOps.h",
        "@heir//include/Dialect/TfheRust/IR:TfheRustTypes.h",
    ],
    deps = [
        "@heir//include/Dialect/TfheRust/IR:dialect_inc_gen",
        "@heir//include/Dialect/TfheRust/IR:ops_inc_gen",
        "@heir//include/Dialect/TfheRust/IR:types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
