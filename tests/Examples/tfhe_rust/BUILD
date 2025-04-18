# See README.md for setup required to run these tests

load("@heir//tests/Examples/tfhe_rust:test.bzl", "tfhe_rs_end_to_end_test")
load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = [
        "Cargo.toml",
        "src/main.rs",
        "src/main_add_one.rs",
        "src/main_fully_connected.rs",
        "src/main_multi_output.rs",
        "src/main_sbox.rs",
        "@heir//tests:test_utilities",
    ],
    default_tags = [
        "manual",
        "notap",
    ],
    driver = "@heir//tests:run_lit.sh",
    size_override = {
        "test_add.mlir": "large",
        "add_one.mlir": "large",
        "test_bitand.mlir": "large",
        "test_simple_lut.mlir": "large",
        "test_fully_connected.mlir": "large",
        "test_sbox.mlir": "large",
        "test_multi_output.mlir": "large",
    },
    test_file_exts = ["mlir"],
)

tfhe_rs_end_to_end_test(
    name = "add_one_test",
    heir_opt_flags = [
        "--forward-store-to-load",
        "--cggi-to-tfhe-rust",
        "--canonicalize",
        "--cse",
    ],
    heir_translate_flags = ["--use-levels"],
    mlir_src = "add_one.mlir",
    test_src = "src/main_add_one.rs",
)
