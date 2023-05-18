# HEIR dialect

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = [
        "@heir//:internal",
    ],
)

cc_library(
    name = "Dialect",
    srcs = [
        "HEIROps.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/HEIR/IR:HEIRDialect.h",
        "@heir//include/Dialect/HEIR/IR:HEIROps.h",
    ],
    includes = ["@heir//include"],
    deps = [
        "@heir//include/Dialect/HEIR/IR:dialect_inc_gen",
        "@heir//include/Dialect/HEIR/IR:ops_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
