package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Patterns",
    srcs = ["Patterns.cpp"],
    hdrs = ["Patterns.h"],
    deps = [
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Transforms/PropagateAnnotation",
        "@heir//lib/Utils",
        "@heir//lib/Utils:AttributeUtils",
        "@heir//lib/Utils:ContextAwareDialectConversion",
        "@heir//lib/Utils:ContextAwareTypeConversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
