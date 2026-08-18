load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "conv2d_nchw_channel_padding",
    go_library_name = "conv2dnchwchannelpadding",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=4096 scaling-mod-bits=45 first-mod-bits=60",
        "--scheme-to-lattigo",
    ],
    mlir_src = "conv2d_nchw_channel_padding.mlir",
)

go_test(
    name = "conv2d_nchw_channel_padding_test",
    srcs = ["conv2d_nchw_channel_padding_test.go"],
    embed = [":conv2dnchwchannelpadding"],
    deps = [":conv2dnchwchannelpadding_utils"],
)
