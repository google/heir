load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "pad_matmul",
    go_library_name = "padmatmul",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=2048",
        "--scheme-to-lattigo",
    ],
    mlir_src = "pad_matmul.mlir",
)

go_test(
    name = "padmatmul_test",
    timeout = "eternal",
    srcs = ["pad_matmul_test.go"],
    embed = [":padmatmul"],
)
