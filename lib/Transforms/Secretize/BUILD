package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Secretize",
    srcs = [
        "Secretize.cpp",
        "WrapGeneric.cpp",
    ],
    hdrs = [
        "@heir//include/Transforms/Secretize:Passes.h",
    ],
    deps = [
        "@heir//include/Transforms/Secretize:pass_inc_gen",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)
