load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

llvm_runner_test(
    name = "lower_add",
    heir_opt_flags = [
        "--mod-arith-to-arith",
        "--emit-c-interface",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "lower_add_test.cc",
    mlir_src = "lower_add.mlir",
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
)
