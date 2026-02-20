load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "mnist",
    go_library_name = "mnist",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common/mnist:mnist.mlir",
)

go_test(
    name = "mnist_test",
    srcs = ["mnist_test.go"],
    data = [
        "@heir//tests/Examples/common/mnist/data:t10k-images-idx3-ubyte",
        "@heir//tests/Examples/common/mnist/data:t10k-labels-idx1-ubyte",
        "@heir//tests/Examples/common/mnist/data:traced_model.pt",
    ],
    embed = [":mnist"],
    deps = [
        "@com_github_tuneinsight_lattigo_v6//core/rlwe",
        "@com_github_tuneinsight_lattigo_v6//schemes/ckks",
    ],
    size = "large",
)


heir_lattigo_lib(
    name = "mnist_layer1",
    go_library_name = "mnist_layer1",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=1024 scaling-mod-bits=45",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common/mnist:mnist_layer1.mlir",
)

go_test(
    name = "mnist_test_layer1",
    srcs = ["mnist_test_layer1.go"],
    data = [
        "@heir//tests/Examples/common/mnist/data:t10k-images-idx3-ubyte",
        "@heir//tests/Examples/common/mnist/data:t10k-labels-idx1-ubyte",
        "@heir//tests/Examples/common/mnist/data:traced_model.pt",
    ],
    embed = [":mnist_layer1"],
    deps = [
        "@com_github_tuneinsight_lattigo_v6//core/rlwe",
        "@com_github_tuneinsight_lattigo_v6//schemes/ckks",
    ],
    size = "large",
)
