load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "chebyshev",
    go_library_name = "chebyshev",
    heir_opt_flags = [
        "--scheme-to-lattigo",
    ],
    mlir_src = ":chebyshev.mlir",
)

go_test(
    name = "chebyshev_test",
    srcs = ["chebyshev_test.go"],
    embed = [":chebyshev"],
    deps = [
        "@com_github_tuneinsight_lattigo_v6//core/rlwe",
        "@com_github_tuneinsight_lattigo_v6//schemes/ckks",
    ],
)
