# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "mnist_test",
    data = [
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",
        "data/traced_model.pt",
    ],
    generated_lib_header = "mnist_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=1024",
        "--scheme-to-openfhe",
    ],
    heir_translate_flags = [
        "--openfhe-include-type=source-relative",
    ],
    mlir_src = "@heir//tests/Examples/common:mnist.mlir",
    tags = [
        "manual",
        "notap",
    ],
    test_src = "mnist_test.cpp",
    deps = [
        "@bazel_tools//tools/cpp/runfiles",
        "@libtorch",
    ],
)
