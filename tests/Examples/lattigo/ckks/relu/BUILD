load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "relu",
    go_library_name = "relu",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=8192",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/lattigo/ckks/relu:relu.mlir",
)

go_test(
    name = "relu_test",
    srcs = ["relu_test.go"],
    embed = [":relu"],
)

