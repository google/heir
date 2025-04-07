# YosysOptimizer pass

load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

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
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
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
    tags = ["yosys"],
    deps = [
        ":LUTImporter",
        ":RTLILImporter",
        "@at_clifford_yosys//:kernel",
        "@at_clifford_yosys//:version",  # buildcleaner: keep
        "@bazel_tools//tools/cpp/runfiles",
        "@googletest//:gtest",
        "@heir//lib/Dialect/Comb/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "BooleanGateImporter",
    srcs = ["BooleanGateImporter.cpp"],
    hdrs = ["BooleanGateImporter.h"],
    deps = [
        ":RTLILImporter",
        "@at_clifford_yosys//:kernel",
        "@heir//lib/Dialect/Comb/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "YosysOptimizer",
    srcs = ["YosysOptimizer.cpp"],
    hdrs = [
        "YosysOptimizer.h",
    ],
    data = [
        "@edu_berkeley_abc//:abc",
        "@heir//lib/Transforms/YosysOptimizer/yosys:share_files",
    ],
    deps = [
        ":BooleanGateImporter",
        ":LUTImporter",
        ":RTLILImporter",
        ":pass_inc_gen",
        "@at_clifford_yosys//:kernel",  # buildcleaner: keep
        "@at_clifford_yosys//:version",  # buildcleaner: keep
        "@heir//lib/Dialect/Comb/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:SecretPatterns",
        "@heir//lib/Target/Verilog:VerilogEmitter",
        "@heir//lib/Utils/RewriteUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "YosysOptimizer",
)
