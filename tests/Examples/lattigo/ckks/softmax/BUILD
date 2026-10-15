load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "softmax",
    go_library_name = "softmax",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--softmax-to-cgf-softmax",
        "--torch-linalg-to-ckks=min-slot-count=1024 first-mod-bits=60 greedy-level-budget=20",
        "--scheme-to-lattigo",
    ],
    mlir_src = "softmax.mlir",
)

go_test(
    name = "softmax_test",
    srcs = ["softmax_test.go"],
    embed = [":softmax"],
)
