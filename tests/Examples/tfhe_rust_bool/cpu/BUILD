load("@heir//tests/Examples/tfhe_rust:test.bzl", "tfhe_rs_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

tfhe_rs_end_to_end_test(
    name = "and_test",
    size = "large",
    heir_translate_flags = ["--emit-tfhe-rust-bool"],
    mlir_src = "test_and.mlir",
    test_src = "src/main.rs",
)

tfhe_rs_end_to_end_test(
    name = "bool_add_test",
    size = "large",
    heir_translate_flags = ["--emit-tfhe-rust-bool"],
    mlir_src = "test_bool_add.mlir",
    test_src = "src/main_bool_add.rs",
)
