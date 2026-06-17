load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "conv2d_dilated",
    go_library_name = "conv2d_dilated",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=1024 scaling-mod-bits=45 first-mod-bits=60 split-preprocessing=1",
        "--scheme-to-lattigo",
    ],
    mlir_src = "conv2d_dilated.mlir",
    split_preprocessing = True,
)

go_test(
    name = "conv2d_dilated_test",
    srcs = ["conv2d_dilated_test.go"],
    embed = [":conv2d_dilated"],
    deps = [":conv2d_dilated_utils"],
)
