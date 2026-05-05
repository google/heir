load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "pooling",
    go_library_name = "pooling",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=4096 scaling-mod-bits=55 first-mod-bits=60 split-preprocessing=1",
        "--scheme-to-lattigo",
    ],
    mlir_src = "pooling.mlir",
    split_preprocessing = True,
)

go_test(
    name = "pooling_test",
    srcs = ["pooling_test.go"],
    embed = [":pooling"],
    deps = [":pooling_utils"],
)
