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
        "manual",
        "notap",
    ],
    test_src = "binops_test.cpp",
)
