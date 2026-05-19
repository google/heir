load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "mnist",
    go_library_name = "mnist",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        # TODO(#2960): remove split-preprocessing=0 once split-preprocessing handles loops
        "--torch-linalg-to-ckks=ciphertext-degree=1024 modulus-switch-after-mul=true experimental-disable-loop-unroll=true level-budget=40 first-mod-bits=55 split-preprocessing=0",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common/mnist:mnist.mlir",
    split_preprocessing = False,
)

go_test(
    name = "mnist_test",
    size = "large",
    srcs = ["mnist_test.go"],
    data = [
        "@heir//tests/Examples/common/mnist/data:t10k-images-idx3-ubyte",
        "@heir//tests/Examples/common/mnist/data:t10k-labels-idx1-ubyte",
        "@heir//tests/Examples/common/mnist/data:traced_model.pt",
    ],
    embed = [":mnist_local"],
)
