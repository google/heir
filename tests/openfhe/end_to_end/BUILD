# See README.md for setup required to run these tests

load("@heir//tests/openfhe/end_to_end:test.bzl", "openfhe_end_to_end_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

openfhe_end_to_end_test(
    name = "binops_test",
    generated_lib_header = "binops_lib.h",
    mlir_src = "binops.mlir",
    tags = [
        "notap",
    ],
    test_src = "binops_test.cpp",
)

openfhe_end_to_end_test(
    name = "simple_sum_test",
    generated_lib_header = "simple_sum_lib.h",
    heir_opt_flags = "--mlir-to-openfhe-bgv=entry-function=simple_sum ciphertext-degree=32",
    mlir_src = "simple_sum.mlir",
    tags = [
        "notap",
    ],
    test_src = "simple_sum_test.cpp",
)
