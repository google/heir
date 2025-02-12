# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

# BGV

openfhe_end_to_end_test(
    name = "binops_test",
    generated_lib_header = "binops_lib.h",
    mlir_src = "binops.mlir",
    tags = ["notap"],
    test_src = "binops_test.cpp",
)

openfhe_end_to_end_test(
    name = "ciphertext_plaintext_ops_test",
    generated_lib_header = "ciphertext_plaintext_ops_lib.h",
    mlir_src = "ciphertext_plaintext_ops.mlir",
    tags = ["notap"],
    test_src = "ciphertext_plaintext_ops_test.cpp",
)

openfhe_end_to_end_test(
    name = "simple_sum_test",
    generated_lib_header = "simple_sum_lib.h",
    heir_opt_flags = [
        "--mlir-to-bgv=ciphertext-degree=32",
        "--scheme-to-openfhe=entry-function=simple_sum",
    ],
    mlir_src = "simple_sum.mlir",
    tags = ["notap"],
    test_src = "simple_sum_test.cpp",
)

openfhe_end_to_end_test(
    name = "dot_product_8_test",
    generated_lib_header = "dot_product_8_lib.h",
    heir_opt_flags = [
        "--mlir-to-bgv=ciphertext-degree=8",
        "--scheme-to-openfhe=entry-function=dot_product",
    ],
    mlir_src = "dot_product_8.mlir",
    tags = ["notap"],
    test_src = "dot_product_8_test.cpp",
)

openfhe_end_to_end_test(
    name = "dot_product_8_debug_test",
    generated_lib_header = "dot_product_8_debug_lib.h",
    heir_opt_flags = [
        "--mlir-to-bgv=ciphertext-degree=8",
        "--scheme-to-openfhe=entry-function=dot_product insert-debug-handler-calls=true",
    ],
    mlir_src = "dot_product_8.mlir",
    tags = ["notap"],
    test_src = "dot_product_8_debug_test.cpp",
)

openfhe_end_to_end_test(
    name = "box_blur_64x64_test",
    generated_lib_header = "box_blur_64x64_lib.h",
    heir_opt_flags = [
        "--mlir-to-bgv=ciphertext-degree=4096 plaintext-modulus=786433",
        "--scheme-to-openfhe=entry-function=box_blur",
    ],
    mlir_src = "box_blur_64x64.mlir",
    tags = ["notap"],
    test_src = "box_blur_test.cpp",
)

openfhe_end_to_end_test(
    name = "roberts_cross_64x64_test",
    generated_lib_header = "roberts_cross_64x64_lib.h",
    heir_opt_flags = [
        "--mlir-to-bgv=ciphertext-degree=4096 plaintext-modulus=536903681",
        "--scheme-to-openfhe=entry-function=roberts_cross",
    ],
    mlir_src = "roberts_cross_64x64.mlir",
    tags = ["notap"],
    test_src = "roberts_cross_test.cpp",
)

# CKKS

openfhe_end_to_end_test(
    name = "dot_product_8f_test",
    generated_lib_header = "dot_product_8f_lib.h",
    heir_opt_flags = [
        "--mlir-to-ckks=ciphertext-degree=8",
        "--scheme-to-openfhe=entry-function=dot_product",
    ],
    heir_translate_flags = [],
    mlir_src = "dot_product_8f.mlir",
    tags = ["notap"],
    test_src = "dot_product_8f_test.cpp",
)

openfhe_end_to_end_test(
    name = "naive_matmul_test",
    generated_lib_header = "naive_matmul_lib.h",
    # We manually specify the passes since the test file is already expressed at the CKKS dialect level.
    heir_opt_flags = [
        "--full-loop-unroll",
        "--ckks-to-lwe",
        "--lwe-to-openfhe",
        "--canonicalize",
        "--cse",
        "--openfhe-configure-crypto-context",
    ],
    heir_translate_flags = [
        "--openfhe-include-type=source-relative",
    ],
    mlir_src = "naive_matmul.mlir",
    tags = ["notap"],
    test_src = "naive_matmul_test.cpp",
)

openfhe_end_to_end_test(
    name = "halevi_shoup_matmul_test",
    generated_lib_header = "halevi_shoup_matmul_lib.h",
    heir_opt_flags = [
        "--mlir-to-ckks=ciphertext-degree=16",
        "--scheme-to-openfhe=entry-function=matmul",
    ],
    heir_translate_flags = [
        "--openfhe-include-type=source-relative",
    ],
    mlir_src = "halevi_shoup_matmul.mlir",
    tags = ["notap"],
    test_src = "halevi_shoup_matmul_test.cpp",
)

openfhe_end_to_end_test(
    name = "simple_ckks_bootstrapping_test",
    generated_lib_header = "simple_ckks_bootstrapping_lib.h",
    heir_opt_flags = [
        "--openfhe-configure-crypto-context=entry-function=simple_ckks_bootstrapping insecure=true",
    ],
    heir_translate_flags = [
        "--openfhe-include-type=source-relative",
    ],
    mlir_src = "simple_ckks_bootstrapping.mlir",
    tags = ["notap"],
    test_src = "simple_ckks_bootstrapping_test.cpp",
)
