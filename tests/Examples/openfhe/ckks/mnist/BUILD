# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "mnist_test",
    generated_lib_header = "mnist_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=16",
        "--scheme-to-openfhe=insert-debug-handler-calls=true",
    ],
    heir_translate_flags = [
        "--openfhe-include-type=source-relative",
    ],
    mlir_src = ":mnist.mlir",
    tags = ["notap"],
    test_src = "mnist_test.cpp",
)
