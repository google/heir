load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "mont_arith",
    hdrs = ["MontArith.h"],
    deps = [],
)

cc_test(
    name = "MontArithTest",
    srcs = ["MontArithTest.cpp"],
    deps = [
        ":mont_arith",
        "@googletest//:gtest_main",
        "@heir//lib/Utils:APIntUtils",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "params",
    srcs = ["Params.cpp"],
    hdrs = ["Params.h"],
    deps = [
        ":mont_arith",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "tables",
    srcs = ["Tables.cpp"],
    hdrs = ["Tables.h"],
    deps = [
        ":mont_arith",
        ":params",
        "@heir//lib/Utils:APIntUtils",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "ref_kernels",
    srcs = ["RefKernels.cpp"],
    hdrs = ["RefKernels.h"],
    deps = [
        ":mont_arith",
        ":tables",
    ],
)

cc_library(
    name = "ref_bconv",
    srcs = ["RefBConv.cpp"],
    hdrs = ["RefBConv.h"],
    deps = [
        ":mont_arith",
        ":tables",
    ],
)

cc_library(
    name = "mod_switch_tables",
    srcs = ["ModSwitchTables.cpp"],
    hdrs = ["ModSwitchTables.h"],
    deps = [
        ":mont_arith",
        ":params",
        ":tables",
    ],
)

cc_library(
    name = "ref_mod_switch",
    srcs = ["RefModSwitch.cpp"],
    hdrs = ["RefModSwitch.h"],
    deps = [
        ":mod_switch_tables",
        ":mont_arith",
        ":params",
        ":ref_bconv",
        ":ref_kernels",
        ":tables",
    ],
)

cc_library(
    name = "ref_key_switch",
    srcs = ["RefKeySwitch.cpp"],
    hdrs = ["RefKeySwitch.h"],
    deps = [
        ":mod_switch_tables",
        ":mont_arith",
        ":params",
        ":ref_kernels",
        ":ref_mod_switch",
        ":tables",
    ],
)

cc_library(
    name = "encoder",
    srcs = ["Encoder.cpp"],
    hdrs = ["Encoder.h"],
    copts = [
        "-fno-sanitize=undefined",
    ],
    deps = [
        ":mont_arith",
        ":params",
        ":ref_kernels",
        ":tables",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "boot_consts",
    srcs = ["BootConsts.cpp"],
    hdrs = ["BootConsts.h"],
    copts = [
        "-fno-sanitize=undefined",
    ],
    deps = [
        ":encoder",
        ":params",
        ":tables",
    ],
)
