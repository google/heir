# BinFHE end-to-end tests

load("@heir//tests/Examples/openfhe/binfhe:test.bzl", "openfhe_binfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

# Split each function into its own end-to-end test to keep runtime bounded.

openfhe_binfhe_end_to_end_test(
    name = "xor_bits_test",
    generated_lib_header = "xor_bits_only_lib.h",
    heir_opt_flags = ["--cggi-to-openfhe"],
    mlir_src = "xor_bits_only.mlir",
    tags = ["notap"],
    test_src = "xor_bits_only_test.cpp",
)

openfhe_binfhe_end_to_end_test(
    name = "and_bits_test",
    generated_lib_header = "and_bits_only_lib.h",
    heir_opt_flags = ["--cggi-to-openfhe"],
    mlir_src = "and_bits_only.mlir",
    tags = ["notap"],
    test_src = "and_bits_only_test.cpp",
)

openfhe_binfhe_end_to_end_test(
    name = "or_bits_test",
    generated_lib_header = "or_bits_only_lib.h",
    heir_opt_flags = ["--cggi-to-openfhe"],
    mlir_src = "or_bits_only.mlir",
    tags = ["notap"],
    test_src = "or_bits_only_test.cpp",
)

openfhe_binfhe_end_to_end_test(
    name = "not_bit_test",
    generated_lib_header = "not_bit_only_lib.h",
    heir_opt_flags = ["--cggi-to-openfhe"],
    mlir_src = "not_bit_only.mlir",
    tags = ["notap"],
    test_src = "not_bit_only_test.cpp",
)

openfhe_binfhe_end_to_end_test(
    name = "full_adder_sum_test",
    generated_lib_header = "full_adder_sum_lib.h",
    heir_opt_flags = ["--cggi-to-openfhe"],
    mlir_src = "full_adder_sum_only.mlir",
    tags = ["notap"],
    test_src = "full_adder_sum_test.cpp",
)

openfhe_binfhe_end_to_end_test(
    name = "full_adder_carry_test",
    generated_lib_header = "full_adder_carry_lib.h",
    heir_opt_flags = ["--cggi-to-openfhe"],
    mlir_src = "full_adder_carry_only.mlir",
    tags = ["notap"],
    test_src = "full_adder_carry_test.cpp",
)
