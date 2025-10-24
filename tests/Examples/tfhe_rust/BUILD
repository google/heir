load("@heir//tests/Examples/tfhe_rust:test.bzl", "tfhe_rs_end_to_end_test")
load("@heir//tools:heir-tfhe-rs.bzl", "tfhe_rs_lib")
load("@rules_rust//rust:defs.bzl", "rust_binary")

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
    tags = [
        "manual",
        "notap",
        "yosys",
    ],
    test_src = "src/main_fully_connected.rs",
)

tfhe_rs_end_to_end_test(
    name = "sbox_test",
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
    tags = [
        "manual",
        "notap",
        "yosys",
    ],
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
    tags = [
        "manual",
        "notap",
        "yosys",
    ],
    test_src = "src/main_multi_output.rs",
)

tfhe_rs_lib(
    name = "add_round_key",
    heir_opt_flags = [
        "--mlir-to-cggi",
        "--scheme-to-tfhe-rs",
    ],
    heir_translate_flags = [
        "--emit-tfhe-rust",
        "--use-levels",
    ],
    mlir_src = "@heir//tests/Examples/common/aes:add_round_key.mlir",
    rs_lib_target_name = "add_round_key_rs_lib",
    tags = [
        "manual",
        "notap",
        "yosys",
    ],
)

tfhe_rs_lib(
    name = "mix_columns",
    heir_opt_flags = [
        "--convert-if-to-select",
        "--full-loop-unroll",
        "--inline",
        "--symbol-dce",
        "--mlir-to-cggi",
        "--scheme-to-tfhe-rs",
    ],
    heir_translate_flags = [
        "--emit-tfhe-rust",
        "--use-levels",
    ],
    mlir_src = "@heir//tests/Examples/common/aes:mix_columns.mlir",
    rs_lib_target_name = "mix_columns_rs_lib",
    tags = [
        "manual",
        "notap",
        "yosys",
    ],
)

tfhe_rs_lib(
    name = "inv_mix_columns",
    heir_opt_flags = [
        "--convert-if-to-select",
        "--full-loop-unroll",
        "--inline",
        "--symbol-dce",
        "--mlir-to-cggi",
        "--scheme-to-tfhe-rs",
    ],
    heir_translate_flags = [
        "--emit-tfhe-rust",
        "--use-levels",
    ],
    mlir_src = "@heir//tests/Examples/common/aes:inv_mix_columns.mlir",
    rs_lib_target_name = "inv_mix_columns_rs_lib",
    tags = [
        "manual",
        "notap",
        "yosys",
    ],
)

tfhe_rs_lib(
    name = "sub_bytes",
    heir_opt_flags = [
        "--inline",
        "--symbol-dce",
        "--mlir-to-cggi",
        "--scheme-to-tfhe-rs",
    ],
    heir_translate_flags = [
        "--emit-tfhe-rust",
        "--use-levels",
    ],
    mlir_src = "@heir//tests/Examples/common/aes:sub_bytes.mlir",
    rs_lib_target_name = "sub_bytes_rs_lib",
    tags = [
        "manual",
        "notap",
        "yosys",
    ],
)

tfhe_rs_lib(
    name = "inv_sub_bytes",
    heir_opt_flags = [
        "--mlir-to-cggi",
        "--scheme-to-tfhe-rs",
    ],
    heir_translate_flags = [
        "--emit-tfhe-rust",
        "--use-levels",
    ],
    mlir_src = "@heir//tests/Examples/common/aes:inv_sub_bytes.mlir",
    rs_lib_target_name = "inv_sub_bytes_rs_lib",
    tags = [
        "manual",
        "notap",
        "yosys",
    ],
)

tfhe_rs_lib(
    name = "shift_rows",
    heir_opt_flags = [
        "--mlir-to-cggi",
        "--scheme-to-tfhe-rs",
    ],
    heir_translate_flags = [
        "--emit-tfhe-rust",
        "--use-levels",
    ],
    mlir_src = "@heir//tests/Examples/common/aes:shift_rows.mlir",
    rs_lib_target_name = "shift_rows_rs_lib",
    tags = [
        "manual",
        "notap",
        "yosys",
    ],
)

tfhe_rs_lib(
    name = "inv_shift_rows",
    heir_opt_flags = [
        "--mlir-to-cggi",
        "--scheme-to-tfhe-rs",
    ],
    heir_translate_flags = [
        "--emit-tfhe-rust",
        "--use-levels",
    ],
    mlir_src = "@heir//tests/Examples/common/aes:inv_shift_rows.mlir",
    rs_lib_target_name = "inv_shift_rows_rs_lib",
    tags = [
        "manual",
        "notap",
        "yosys",
    ],
)

rust_binary(
    name = "aes",
    srcs = ["src/main_aes.rs"],
    tags = [
        "manual",
        "notap",
        "yosys",
    ],
    deps = [
        ":add_round_key_rs_lib",
        ":inv_mix_columns_rs_lib",
        ":inv_shift_rows_rs_lib",
        ":inv_sub_bytes_rs_lib",
        ":mix_columns_rs_lib",
        ":shift_rows_rs_lib",
        ":sub_bytes_rs_lib",
        "@crates//:rayon",
        "@crates//:serde",
        "@crates//:tfhe",
    ],
)
