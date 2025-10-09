# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "box_blur_16x16_test",
    generated_lib_header = "box_blur_16x16_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=bgv",
        "--mlir-to-bgv=plaintext-modulus=786433",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/common:box_blur_16x16.mlir",
    test_src = "box_blur_test.cpp",
)
