package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "CombDialect.cpp",
        "CombOps.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Comb/IR:CombDialect.h",
        "@heir//include/Dialect/Comb/IR:CombOps.h",
    ],
    deps = [
        "@heir//include/Dialect/Comb/IR:dialect_inc_gen",
        "@heir//include/Dialect/Comb/IR:enum_inc_gen",
        "@heir//include/Dialect/Comb/IR:ops_inc_gen",
        "@heir//include/Dialect/Comb/IR:type_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)
