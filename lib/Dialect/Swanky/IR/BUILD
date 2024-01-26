# Swanky dialect implementation

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "SwankyDialect.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/Swanky/IR:SwankyDialect.h",
    ],
    deps = [
        "@heir//include/Dialect/Swanky/IR:dialect_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
