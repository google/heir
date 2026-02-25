load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "dot_product_8_default_debug_test",
    generate_debug_helper = True,
    generated_lib_header = "dot_product_8_debug_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=bgv",
        "--mlir-to-bfv=ciphertext-degree=8192 annotate-noise-bound=true",
        "--scheme-to-openfhe=insert-debug-handler-calls=true",
    ],
    mlir_src = "@heir//tests/Examples/common:dot_product_8.mlir",
    test_src = "dot_product_8_default_debug_test.cpp",
)
