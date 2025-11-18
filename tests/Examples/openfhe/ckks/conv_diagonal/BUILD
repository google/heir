load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_interpreter_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_interpreter_test(
    name = "conv_layer_interpreter_test",
    copts = select({
        "@heir//:config_openfhe_enable_timing": ["-DOPENFHE_ENABLE_TIMING"],
        "//conditions:default": [],
    }),
    generated_heir_opt_filename = "module.openfhe.mlir",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=2048",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/openfhe/ckks/conv_diagonal:conv_layer.mlir",
    test_src = "conv_layer_interpreter_test.cpp",
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)
