load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "lola",
    go_library_name = "lola",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=2048",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:lola.mlir",
    split_preprocessing = True,
)

go_test(
    name = "lola_test",
    size = "large",
    srcs = ["lola_test.go"],
    data = [
        "@heir//tests/Examples/common/mnist/data:t10k-images-idx3-ubyte",
    ],
    embed = [":lola"],
    pure = "on",
    deps = [":lola_utils"],
)
