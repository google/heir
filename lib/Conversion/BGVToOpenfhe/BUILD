package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "BGVToOpenfhe",
    srcs = ["BGVToOpenfhe.cpp"],
    hdrs = [
        "@heir//include/Conversion/BGVToOpenfhe:BGVToOpenfhe.h",
    ],
    deps = [
        "@heir//include/Conversion/BGVToOpenfhe:pass_inc_gen",
        "@heir//lib/Conversion:Utils",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)
