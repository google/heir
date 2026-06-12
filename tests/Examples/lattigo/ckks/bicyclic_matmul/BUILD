load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "bicyclic_matmul",
    go_library_name = "bicyclicmatmul",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=2048 split-preprocessing=0",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:bicyclic_matmul.mlir",
    split_preprocessing = False,
)

go_test(
    name = "bicyclic_matmul_test",
    srcs = ["bicyclic_matmul_test.go"],
    embed = [":bicyclicmatmul"],
)
