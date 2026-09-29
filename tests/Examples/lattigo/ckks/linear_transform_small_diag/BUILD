load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "linear_transform_small_diag",
    go_library_name = "lineartransformsmalldiag",
    heir_opt_flags = [
        "--lattigo-configure-crypto-context=entry-function=test_small_diag",
    ],
    mlir_src = "linear_transform_small_diag.mlir",
    split_preprocessing = False,
)

go_test(
    name = "linear_transform_small_diag_test",
    srcs = ["linear_transform_small_diag_test.go"],
    embed = [":lineartransformsmalldiag"],
    deps = [
        "@com_github_tuneinsight_lattigo_v6//schemes/ckks",
    ],
)
