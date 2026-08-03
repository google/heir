load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "ext_const_test",
    generated_lib_header = "ext_const_lib.h",
    heir_opt_flags = [
        "--ext-const-threshold=2",
    ],
    mlir_src = "ext_const.mlir",
    test_src = "ext_const_test.cpp",
)
