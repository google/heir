load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "conv2d_nchw_padded",
    go_library_name = "conv2dnchwpadded",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=min-slot-count=1024 greedy-modulus-switch-after-mul=true",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:conv2d_nchw_padded.mlir",
)

go_test(
    name = "conv2dnchwpadded_test",
    size = "small",
    srcs = ["conv2d_nchw_padded_test.go"],
    embed = [":conv2dnchwpadded"],
)
