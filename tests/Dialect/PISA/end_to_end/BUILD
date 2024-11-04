# See README.md for setup required to run these tests

load("@heir//tests/Dialect/PISA/end_to_end:test.bzl", "pisa_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

pisa_end_to_end_test(
    name = "basic",
    generated_lib_header = "basic_inc.h",
    heir_opt_flags = "--mlir-to-openfhe-bgv=entry-function=basic_test ciphertext-degree=8192",
    mlir_src = "basic.mlir",
    tags = ["notap"],
    test_src = "basic.cpp",
)
