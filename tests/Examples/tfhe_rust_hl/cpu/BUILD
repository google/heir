load("@heir//bazel:yosys.bzl", "requires_yosys")
load("@heir//tests/Examples/tfhe_rust:test.bzl", "tfhe_rs_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

tfhe_rs_end_to_end_test(
    name = "arith_test",
    size = "large",
    heir_translate_flags = ["--emit-tfhe-rust-hl"],
    mlir_src = "arith.mlir",
    test_src = "src/main_arith.rs",
)

tfhe_rs_end_to_end_test(
    name = "fully_connected_test",
    size = "large",
    heir_opt_flags = [
        "--mlir-to-cggi=data-type=Integer",
        "--scheme-to-tfhe-rs",
    ],
    heir_translate_flags = ["--emit-tfhe-rust-hl"],
    mlir_src = "fully_connected.mlir",
    test_src = "src/main_fully_connected.rs",
)

tfhe_rs_end_to_end_test(
    name = "add_round_key_test",
    size = "large",
    heir_opt_flags = [
        "--mlir-to-cggi=data-type=Integer",
        "--scheme-to-tfhe-rs",
    ],
    heir_translate_flags = ["--emit-tfhe-rust-hl"],
    mlir_src = "add_round_key.mlir",
    # TODO: the no-yosys --mlir-to-cggi=data-type=Integer pipeline produces IR
    # for this input that crashes --emit-tfhe-rust-hl (OpOperand index
    # assertion); only built with yosys until that is fixed.
    target_compatible_with = requires_yosys(),
    test_src = "src/main_add_round_key.rs",
)

tfhe_rs_end_to_end_test(
    name = "hello_world_clean_xsmall_test",
    size = "large",
    heir_opt_flags = [
        "--mlir-to-cggi=data-type=Integer",
        "--scheme-to-tfhe-rs",
    ],
    heir_translate_flags = ["--emit-tfhe-rust-hl"],
    mlir_src = "hello_world_clean_xsmall.mlir",
    # TODO: the no-yosys --mlir-to-cggi=data-type=Integer pipeline produces IR
    # for this input that crashes --emit-tfhe-rust-hl (OpOperand index
    # assertion); only built with yosys until that is fixed.
    target_compatible_with = requires_yosys(),
    test_src = "src/main_hello_world.rs",
)
