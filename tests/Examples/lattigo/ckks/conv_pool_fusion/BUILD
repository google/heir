load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "conv_pool_fusion",
    go_library_name = "conv_pool_fusion",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=4096 modulus-switch-after-mul=true experimental-disable-loop-unroll=true",
        "--scheme-to-lattigo",
    ],
    mlir_src = "conv_pool_fusion.mlir",
    split_preprocessing = True,
)

go_test(
    name = "conv_pool_fusion_test",
    srcs = ["conv_pool_fusion_test.go"],
    embed = [":conv_pool_fusion"],
    deps = [":conv_pool_fusion_utils"],
)
