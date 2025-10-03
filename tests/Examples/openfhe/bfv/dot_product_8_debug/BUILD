# See README.md for setup required to run these tests

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
    name = "dot_product_8_debug_test",
    generated_lib_header = "dot_product_8_debug_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=bfv",
        "--mlir-to-bfv=ciphertext-degree=8192 annotate-noise-bound=true",
        "--scheme-to-openfhe=insert-debug-handler-calls=true",
    ],
    mlir_src = "@heir//tests/Examples/common:dot_product_8.mlir",
    test_src = "dot_product_8_debug_test.cpp",
    deps = [":debug_helper"],
)
