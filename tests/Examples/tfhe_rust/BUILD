load("@heir//tests/Examples/tfhe_rust:test.bzl", "tfhe_rs_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

tfhe_rs_end_to_end_test(
    name = "add_one_test",
    size = "large",
    heir_opt_flags = [
        "--forward-store-to-load",
        "--cggi-to-tfhe-rust",
        "--canonicalize",
        "--cse",
    ],
    heir_translate_flags = [
        "--emit-tfhe-rust",
        "--use-levels",
    ],
    mlir_src = "add_one.mlir",
    test_src = "src/main_add_one.rs",
)

tfhe_rs_end_to_end_test(
    name = "add_test",
    size = "large",
    heir_translate_flags = ["--emit-tfhe-rust"],
    mlir_src = "add.mlir",
    test_src = "src/main_add.rs",
)

tfhe_rs_end_to_end_test(
    name = "bitand_test",
    size = "large",
    heir_translate_flags = [
        "--emit-tfhe-rust",
        "--use-levels",
    ],
    mlir_src = "bitand.mlir",
    test_src = "src/main_bitand.rs",
)

tfhe_rs_end_to_end_test(
    name = "simple_lut_test",
    size = "large",
    heir_translate_flags = ["--emit-tfhe-rust"],
    mlir_src = "simple_lut.mlir",
    test_src = "src/main_simple_lut.rs",
)

tfhe_rs_end_to_end_test(
    name = "fully_connected_test",
    size = "large",
    heir_opt_flags = [
        "--mlir-to-cggi",
        "--scheme-to-tfhe-rs",
    ],
    heir_translate_flags = [
        "--emit-tfhe-rust",
        "--use-levels",
    ],
    mlir_src = "fully_connected.mlir",
    tags = ["yosys"],
    test_src = "src/main_fully_connected.rs",
)

tfhe_rs_end_to_end_test(
    name = "sbox_test",
    # Skip in debug mode
    size = "enormous",
    heir_opt_flags = [
        "--mlir-to-cggi",
        "--scheme-to-tfhe-rs",
    ],
    heir_translate_flags = [
        "--emit-tfhe-rust",
        "--use-levels",
    ],
    mlir_src = "sbox.mlir",
    tags = ["yosys"],
    test_src = "src/main_sbox.rs",
)

tfhe_rs_end_to_end_test(
    name = "multi_output_test",
    size = "large",
    heir_opt_flags = [
        "--mlir-to-cggi",
        "--scheme-to-tfhe-rs",
    ],
    heir_translate_flags = [
        "--emit-tfhe-rust",
        "--use-levels",
    ],
    mlir_src = "multi_output.mlir",
    tags = ["yosys"],
    test_src = "src/main_multi_output.rs",
)
