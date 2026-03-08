load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "pooling",
    go_library_name = "pooling",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/lattigo/ckks/pooling:pooling.mlir",
)

go_test(
    name = "pooling_test",
    srcs = ["pooling_test.go"],
    embed = [":pooling"],
)
