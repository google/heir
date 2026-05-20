# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "in_place",
    go_library_name = "main",
    heir_opt_flags = [
        "--canonicalize",
        "--cse",
        "--scheme-to-lattigo",
    ],
    mlir_src = "in_place.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.

go_test(
    name = "in_place_test",
    srcs = ["in_place_test.go"],
    embed = [":main"],
    deps = [
        "@com_github_tuneinsight_lattigo_v6//core/rlwe",
        "@com_github_tuneinsight_lattigo_v6//schemes/ckks",
    ],
)
