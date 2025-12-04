load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "linear_transform",
    go_library_name = "lineartransform",
    heir_opt_flags = [
        "--scheme-to-lattigo",
    ],
    mlir_src = ":linear_transform.mlir",
)

go_test(
    name = "linear_transform_test",
    srcs = ["linear_transform_test.go"],
    embed = [":lineartransform"],
    deps = [
        "@com_github_tuneinsight_lattigo_v6//circuits/ckks/lintrans",
        "@com_github_tuneinsight_lattigo_v6//core/rlwe",
        "@com_github_tuneinsight_lattigo_v6//ring",
        "@com_github_tuneinsight_lattigo_v6//schemes/ckks",
    ],
)
