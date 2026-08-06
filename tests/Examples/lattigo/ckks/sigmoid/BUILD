load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "sigmoid",
    go_library_name = "sigmoid",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=2048 scaling-mod-bits=55 first-mod-bits=60",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:sigmoid.mlir",
    split_preprocessing = False,
)

go_test(
    name = "sigmoid_test",
    srcs = ["sigmoid_test.go"],
    embed = [":sigmoid"],
)
