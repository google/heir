# YosysOptimizer pass

load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "RTLILImporter",
    srcs = ["RTLILImporter.cpp"],
    hdrs = ["RTLILImporter.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@yosys//:backend_rtlil",
        "@yosys//:kernel",
        "@yosys//:frontend_liberty",
    ],
)

cc_library(
    name = "LUTImporter",
    srcs = ["LUTImporter.cpp"],
    hdrs = ["LUTImporter.h"],
    deps = [
        ":RTLILImporter",
        "@heir//lib/Dialect/Comb/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@yosys//:kernel",
        "@yosys//:frontend_liberty",
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
        "@bazel_tools//tools/cpp/runfiles",
        "@googletest//:gtest",
        "@heir//lib/Dialect/Comb/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@yosys//:kernel",
    ],
)

cc_library(
    name = "BooleanGateImporter",
    srcs = ["BooleanGateImporter.cpp"],
    hdrs = ["BooleanGateImporter.h"],
    deps = [
        ":RTLILImporter",
        "@heir//lib/Dialect/Comb/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@yosys//:kernel",
        "@yosys//:frontend_liberty",
    ],
)

cc_library(
    name = "YosysOptimizer",
    srcs = ["YosysOptimizer.cpp"],
    hdrs = [
        "YosysOptimizer.h",
    ],
    data = [
        "@abc//:abc_bin",
        "@heir//lib/Transforms/YosysOptimizer/yosys:techmap_lut3.v",
        "@heir//lib/Transforms/YosysOptimizer/yosys:techmap_lut4.v",
    ],
    deps = [
        ":BooleanGateImporter",
        ":LUTImporter",
        ":RTLILImporter",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Comb/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:SecretPatterns",
        "@heir//lib/Target/Verilog:VerilogEmitter",
        "@heir//lib/Utils:TransformUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@yosys//:backend_rtlil",
        "@yosys//:frontend_verilog",
        "@yosys//:kernel",  # buildcleaner: keep
        "@yosys//:pass_cmds",
        "@yosys//:pass_techmap",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "YosysOptimizer",
)
