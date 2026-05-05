load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "conv2d_nchw_tall_1",
    go_library_name = "conv2dnchwtall1",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:conv2d_nchw_tall_1.mlir",
)

go_test(
    name = "conv2dnchwtall1_test",
    size = "small",
    srcs = ["conv2d_nchw_tall_1_test.go"],
    embed = [":conv2dnchwtall1"],
)
