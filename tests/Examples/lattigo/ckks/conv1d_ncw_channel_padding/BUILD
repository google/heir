load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "conv1d_ncw_channel_padding",
    go_library_name = "conv1dncwchannelpadding",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=min-slot-count=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "conv1d_ncw_channel_padding.mlir",
)

go_test(
    name = "conv1d_ncw_channel_padding_test",
    size = "small",
    srcs = ["conv1d_ncw_channel_padding_test.go"],
    embed = [":conv1dncwchannelpadding"],
)
