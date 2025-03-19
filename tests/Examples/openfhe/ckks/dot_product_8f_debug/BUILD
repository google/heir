# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "dot_product_8f_debug_test",
    data = [
        "@heir//tests/Examples/plaintext/dot_product_f_debug:dot_product_8f_debug.log",
    ],
    generated_lib_header = "dot_product_8f_debug_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=512 \
          plaintext-execution-result-file-name=$(location @heir//tests/Examples/plaintext/dot_product_f_debug:dot_product_8f_debug.log)",
        "--scheme-to-openfhe=insert-debug-handler-calls=true",
    ],
    heir_translate_flags = [],
    mlir_src = "@heir//tests/Examples/common:dot_product_8f.mlir",
    tags = ["notap"],
    test_src = "dot_product_8f_debug_test.cpp",
)
