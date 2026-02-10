load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "bootstrap",
    go_library_name = "bootstrap",
    heir_opt_flags = [
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/lattigo/ckks/bootstrapping:bootstrap.mlir",
)

go_test(
    name = "bootstrap_test",
    srcs = ["bootstrap_test.go"],
    embed = [":bootstrap"],
    deps = [
        "@com_github_tuneinsight_lattigo_v6//core/rlwe",
        "@com_github_tuneinsight_lattigo_v6//schemes/ckks",
    ],
)
