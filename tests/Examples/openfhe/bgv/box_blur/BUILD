# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "box_blur_64x64_test",
    generated_lib_header = "box_blur_64x64_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=bgv",
        "--mlir-to-bgv=ciphertext-degree=4096 plaintext-modulus=786433",
        # override the default scheme-to-openfhe because it fails to
        # select adequate parameters as of v1.3.
        # "--scheme-to-openfhe"
        "--bgv-to-lwe",
        "--lwe-to-openfhe",
        "--canonicalize",
        "--cse",
        # ensure the ring dimension is large enough.
        "--openfhe-configure-crypto-context=ring-dim=8192",
    ],
    mlir_src = "@heir//tests/Examples/common:box_blur_64x64.mlir",
    tags = ["notap"],
    test_src = "box_blur_test.cpp",
)
