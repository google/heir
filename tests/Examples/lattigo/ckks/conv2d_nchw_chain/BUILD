load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "conv2d_nchw_chain",
    go_library_name = "conv2dnchwchain",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=4096 scaling-mod-bits=45 first-mod-bits=60",
        "--scheme-to-lattigo",
    ],
    mlir_src = "conv2d_nchw_chain.mlir",
)

go_test(
    name = "conv2d_nchw_chain_test",
    srcs = ["conv2d_nchw_chain_test.go"],
    embed = [":conv2dnchwchain"],
    deps = [":conv2dnchwchain_utils"],
)
