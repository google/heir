# YosysOptimizer pass

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "RTLILImporter",
    srcs = ["RTLILImporter.cpp"],
    hdrs = ["RTLILImporter.h"],
    deps = [
        "@at_clifford_yosys//:kernel",
        "@heir//lib/Dialect/Comb/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "LUTImporter",
    srcs = ["LUTImporter.cpp"],
    hdrs = ["LUTImporter.h"],
    deps = [
        ":RTLILImporter",
        "@at_clifford_yosys//:kernel",
        "@heir//lib/Dialect/Comb/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "LUTImporterTest",
    size = "small",
    srcs = ["LUTImporterTest.cpp"],
    data = glob([
        "tests/*.rtlil",
    ]),
    deps = [
        ":LUTImporter",
        "@at_clifford_yosys//:kernel",
        "@at_clifford_yosys//:version",
        "@bazel_tools//tools/cpp/runfiles",
        "@googletest//:gtest",
        "@heir//lib/Dialect/Comb/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
