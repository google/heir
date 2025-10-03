# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "cmux_test",
    generated_lib_header = "cmux_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=bgv",
        "--mlir-to-bgv",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/common:cmux.mlir",
    test_src = "cmux_test.cpp",
)
