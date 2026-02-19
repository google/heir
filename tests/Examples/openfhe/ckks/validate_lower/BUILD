load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_applicable_licenses = ["@heir//:license"])

cc_library(
    name = "debug_helper",
    srcs = ["debug_helper.cpp"],
    hdrs = ["debug_helper.h"],
    deps = [
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)

openfhe_end_to_end_test(
    name = "validate_lower_test",
    generated_lib_header = "validate_lower_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=8",
        "--scheme-to-openfhe",
    ],
    heir_translate_flags = [
        "--openfhe-debug-helper-include-path=tests/Examples/openfhe/ckks/validate_lower/debug_helper.h",
    ],
    mlir_src = "validate_lower.mlir",
    test_src = "validate_lower_test.cpp",
    deps = [":debug_helper"],
)
