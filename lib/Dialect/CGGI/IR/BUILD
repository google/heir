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
        "@heir//include/Dialect/CGGI/IR:CGGIDialect.h",
        "@heir//include/Dialect/CGGI/IR:CGGIOps.h",
    ],
    includes = ["@heir//include"],
    deps = [
        "@heir//include/Dialect/CGGI/IR:dialect_inc_gen",
        "@heir//include/Dialect/CGGI/IR:ops_inc_gen",
        "@heir//include/Dialect/Poly/IR:attributes_inc_gen",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Poly/IR:PolyAttributes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
