load("@heir//tests/Examples/tfhe_rust:test.bzl", "tfhe_rs_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

tfhe_rs_end_to_end_test(
    name = "arith_test",
    size = "large",
    heir_translate_flags = ["--emit-tfhe-rust-hl"],
    mlir_src = "arith.mlir",
    test_src = "src/main.rs",
)
