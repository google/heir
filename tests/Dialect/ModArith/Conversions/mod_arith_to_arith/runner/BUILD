load("@heir//bazel:lit.bzl", "glob_lit_tests")
load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = [
        "@heir//tests:test_utilities",
        "@llvm-project//clang",
        "@llvm-project//llvm:llc",
        "@llvm-project//mlir:mlir-translate",
    ],
    default_tags = [
        "notap",
        "manual",
    ],
    driver = "@heir//tests:run_lit.sh",
    exclude = [
        "lower_add.mlir",
        "lower_barrett_reduce.mlir",
        "lower_mac.mlir",
        "lower_mod_switch_decompose.mlir",
        "lower_mul.mlir",
        "lower_reduce.mlir",
        "lower_sub.mlir",
        "lower_mod_switch_same_width.mlir",
        "lower_mod_switch_smaller_width.mlir",
        "lower_mod_switch_larger_width.mlir",
        "lower_mod_switch_interpolate.mlir",
    ],
    test_file_exts = ["mlir"],
)

llvm_runner_test(
    name = "lower_add",
    heir_opt_flags = [
        "--mod-arith-to-arith",
        "--emit-c-interface",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "lower_add_test.cc",
    mlir_src = "lower_add.mlir",
    deps = [
        "@googletest//:gtest_main",
        "@heir//tests/llvm_runner:memrefCopy",
    ],
)

llvm_runner_test(
    name = "lower_barrett_reduce",
    heir_opt_flags = [
        "--mod-arith-to-arith",
        "--emit-c-interface",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "lower_barrett_reduce_test.cc",
    mlir_src = "lower_barrett_reduce.mlir",
    deps = [
        "@googletest//:gtest_main",
        "@heir//tests/llvm_runner:memrefCopy",
    ],
)

llvm_runner_test(
    name = "lower_mac",
    heir_opt_flags = [
        "--mod-arith-to-arith",
        "--emit-c-interface",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "lower_mac_test.cc",
    mlir_src = "lower_mac.mlir",
    deps = [
        "@googletest//:gtest_main",
        "@heir//tests/llvm_runner:memrefCopy",
    ],
)

llvm_runner_test(
    name = "lower_mod_switch_decompose",
    heir_opt_flags = [
        "--mod-arith-to-arith",
        "--emit-c-interface",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "lower_mod_switch_decompose_test.cc",
    mlir_src = "lower_mod_switch_decompose.mlir",
    deps = [
        "@googletest//:gtest_main",
        "@heir//tests/llvm_runner:memrefCopy",
    ],
)

llvm_runner_test(
    name = "lower_mul",
    heir_opt_flags = [
        "--mod-arith-to-arith",
        "--emit-c-interface",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "lower_mul_test.cc",
    mlir_src = "lower_mul.mlir",
    deps = [
        "@googletest//:gtest_main",
        "@heir//tests/llvm_runner:memrefCopy",
    ],
)

llvm_runner_test(
    name = "lower_reduce",
    heir_opt_flags = [
        "--mod-arith-to-arith",
        "--emit-c-interface",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "lower_reduce_test.cc",
    mlir_src = "lower_reduce.mlir",
    deps = [
        "@googletest//:gtest_main",
        "@heir//tests/llvm_runner:memrefCopy",
    ],
)

llvm_runner_test(
    name = "lower_sub",
    heir_opt_flags = [
        "--mod-arith-to-arith",
        "--emit-c-interface",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "lower_sub_test.cc",
    mlir_src = "lower_sub.mlir",
    deps = [
        "@googletest//:gtest_main",
        "@heir//tests/llvm_runner:memrefCopy",
    ],
)

llvm_runner_test(
    name = "lower_mod_switch_same_width",
    heir_opt_flags = [
        "--mod-arith-to-arith",
        "--emit-c-interface",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "lower_mod_switch_same_width_test.cc",
    mlir_src = "lower_mod_switch_same_width.mlir",
    deps = [
        "@googletest//:gtest_main",
        "@heir//tests/llvm_runner:memrefCopy",
    ],
)

llvm_runner_test(
    name = "lower_mod_switch_smaller_width",
    heir_opt_flags = [
        "--mod-arith-to-arith",
        "--emit-c-interface",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "lower_mod_switch_smaller_width_test.cc",
    mlir_src = "lower_mod_switch_smaller_width.mlir",
    deps = [
        "@googletest//:gtest_main",
        "@heir//tests/llvm_runner:memrefCopy",
    ],
)

llvm_runner_test(
    name = "lower_mod_switch_larger_width",
    heir_opt_flags = [
        "--mod-arith-to-arith",
        "--emit-c-interface",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "lower_mod_switch_larger_width_test.cc",
    mlir_src = "lower_mod_switch_larger_width.mlir",
    deps = [
        "@googletest//:gtest_main",
        "@heir//tests/llvm_runner:memrefCopy",
    ],
)

llvm_runner_test(
    name = "lower_mod_switch_interpolate",
    heir_opt_flags = [
        "--mod-arith-to-arith",
        "--emit-c-interface",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "lower_mod_switch_interpolate_test.cc",
    mlir_src = "lower_mod_switch_interpolate.mlir",
    deps = [
        "@googletest//:gtest_main",
        "@heir//tests/llvm_runner:memrefCopy",
    ],
)
