load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "batchnorm1d",
    go_library_name = "batchnorm1d",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--linalg-canonicalizations",
        "--torch-linalg-to-ckks=ciphertext-degree=2048 split-preprocessing=1",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:batchnorm1d.mlir",
)

go_test(
    name = "batchnorm1d_test",
    srcs = ["batchnorm1d_test.go"],
    embed = [":batchnorm1d"],
)
