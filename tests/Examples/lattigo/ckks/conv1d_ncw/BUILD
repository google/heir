load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "conv1d_ncw",
    go_library_name = "conv1dncw",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:conv1d_ncw.mlir",
)

go_test(
    name = "conv1dncw_test",
    size = "small",
    srcs = ["conv1d_ncw_test.go"],
    embed = [":conv1dncw"],
)
