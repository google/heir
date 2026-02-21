# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "matvec_512x784_test",
    size = "enormous",
    exec_properties = {"mem": "28g"},
    generated_lib_header = "matvec_512x784_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/common:matvec_512x784.mlir",
    tags = [
        "nofastbuild",
        "notap",
    ],
    test_src = "matvec_512x784_test.cpp",
)
