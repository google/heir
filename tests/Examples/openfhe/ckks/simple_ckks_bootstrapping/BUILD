# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "simple_ckks_bootstrapping_test",
    generated_lib_header = "simple_ckks_bootstrapping_lib.h",
    heir_opt_flags = [
        "--openfhe-configure-crypto-context=insecure=true ring-dim=128",
    ],
    heir_translate_flags = [
        "--openfhe-include-type=source-relative",
    ],
    mlir_src = "simple_ckks_bootstrapping.mlir",
    test_src = "simple_ckks_bootstrapping_test.cpp",
)
