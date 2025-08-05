# Utilities for layout manipulation

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Codegen",
    srcs = ["Codegen.cpp"],
    hdrs = ["Codegen.h"],
    deps = [
        "@isl",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "Parser",
    srcs = ["Parser.cpp"],
    hdrs = ["Parser.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "CodegenTest",
    srcs = ["CodegenTest.cpp"],
    deps = [
        ":Codegen",
        ":Parser",
        "@googletest//:gtest_main",
        "@isl",
    ],
)
