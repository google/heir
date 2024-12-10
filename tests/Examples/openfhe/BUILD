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
    name = "simple_sum_test",
    generated_lib_header = "simple_sum_lib.h",
    heir_opt_flags = ["--mlir-to-openfhe-bgv=entry-function=simple_sum ciphertext-degree=32"],
    mlir_src = "simple_sum.mlir",
    tags = ["notap"],
    test_src = "simple_sum_test.cpp",
)

openfhe_end_to_end_test(
    name = "dot_product_8_test",
    generated_lib_header = "dot_product_8_lib.h",
    heir_opt_flags = ["--mlir-to-openfhe-bgv=entry-function=dot_product ciphertext-degree=8"],
    mlir_src = "dot_product_8.mlir",
    tags = ["notap"],
    test_src = "dot_product_8_test.cpp",
)

openfhe_end_to_end_test(
    name = "box_blur_64x64_test",
    generated_lib_header = "box_blur_64x64_lib.h",
    heir_opt_flags = ["--mlir-to-openfhe-bgv=entry-function=box_blur ciphertext-degree=4096"],
    mlir_src = "box_blur_64x64.mlir",
    tags = ["notap"],
    test_src = "box_blur_test.cpp",
)

openfhe_end_to_end_test(
    name = "roberts_cross_64x64_test",
    generated_lib_header = "roberts_cross_64x64_lib.h",
    heir_opt_flags = ["--mlir-to-openfhe-bgv=entry-function=roberts_cross ciphertext-degree=4096"],
    mlir_src = "roberts_cross_64x64.mlir",
    tags = ["notap"],
    test_src = "roberts_cross_test.cpp",
)

# CKKS

openfhe_end_to_end_test(
    name = "dot_product_8f_test",
    generated_lib_header = "dot_product_8f_lib.h",
    heir_opt_flags = ["--mlir-to-openfhe-ckks=entry-function=dot_product ciphertext-degree=8"],
    heir_translate_flags = ["--openfhe-scheme=ckks"],
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
        "--ckks-to-openfhe",
        "--canonicalize",
        "--cse",
        "--openfhe-configure-crypto-context",
    ],
    heir_translate_flags = [
        "--openfhe-scheme=ckks",
        "--openfhe-include-type=source-relative",
    ],
    mlir_src = "naive_matmul.mlir",
    tags = ["notap"],
    test_src = "naive_matmul_test.cpp",
)

openfhe_end_to_end_test(
    name = "halevi_shoup_matmul_test",
    generated_lib_header = "halevi_shoup_matmul_lib.h",
    heir_opt_flags = ["--mlir-to-openfhe-ckks=entry-function=matmul ciphertext-degree=16"],
    heir_translate_flags = [
        "--openfhe-scheme=ckks",
        "--openfhe-include-type=source-relative",
    ],
    mlir_src = "halevi_shoup_matmul.mlir",
    tags = ["notap"],
    test_src = "halevi_shoup_matmul_test.cpp",
)
