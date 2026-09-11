# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

# TODO(#2257): Loop unrolling is required because convert-to-ciphertext-semantics
# does not support dynamic extraction indices from ciphertexts.
openfhe_end_to_end_test(
    name = "box_blur_16x16_test",
    generated_lib_header = "box_blur_16x16_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=bgv",
        "--mlir-to-bgv=plaintext-modulus=786433 experimental-disable-loop-unroll=false",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/common:box_blur_16x16.mlir",
    tags = ["notap"],
    test_src = "box_blur_test.cpp",
)
