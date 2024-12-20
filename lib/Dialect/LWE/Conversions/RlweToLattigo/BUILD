package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "RlweToLattigo",
    srcs = ["RlweToLattigo.cpp"],
    hdrs = [
        "RlweToLattigo.h",
    ],
    deps = [
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Lattigo/IR:Dialect",
        "@heir//lib/Utils",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)
