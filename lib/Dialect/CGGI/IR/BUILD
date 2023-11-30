package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "CGGIDialect.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/CGGI/IR:CGGIAttributes.h",
        "@heir//include/Dialect/CGGI/IR:CGGIDialect.h",
        "@heir//include/Dialect/CGGI/IR:CGGIOps.h",
    ],
    deps = [
        "@heir//include/Dialect/CGGI/IR:attributes_inc_gen",
        "@heir//include/Dialect/CGGI/IR:dialect_inc_gen",
        "@heir//include/Dialect/CGGI/IR:ops_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
