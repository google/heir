# See README.md for setup required to run these tests

load("@heir//tests/openfhe/end_to_end:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

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
    heir_opt_flags = "--mlir-to-openfhe-bgv=entry-function=simple_sum ciphertext-degree=32",
    mlir_src = "simple_sum.mlir",
    tags = ["notap"],
    test_src = "simple_sum_test.cpp",
)

openfhe_end_to_end_test(
    name = "dot_product_8_test",
    generated_lib_header = "dot_product_8_lib.h",
    heir_opt_flags = "--mlir-to-openfhe-bgv=entry-function=dot_product ciphertext-degree=8",
    mlir_src = "dot_product_8.mlir",
    tags = ["notap"],
    test_src = "dot_product_8_test.cpp",
)

openfhe_end_to_end_test(
    name = "box_blur_64x64_test",
    generated_lib_header = "box_blur_64x64_lib.h",
    heir_opt_flags = "--mlir-to-openfhe-bgv=entry-function=box_blur ciphertext-degree=4096",
    mlir_src = "box_blur_64x64.mlir",
    tags = ["notap"],
    test_src = "box_blur_test.cpp",
)

openfhe_end_to_end_test(
    name = "roberts_cross_64x64_test",
    generated_lib_header = "roberts_cross_64x64_lib.h",
    heir_opt_flags = "--mlir-to-openfhe-bgv=entry-function=roberts_cross ciphertext-degree=4096",
    mlir_src = "roberts_cross_64x64.mlir",
    tags = ["notap"],
    test_src = "roberts_cross_test.cpp",
)
