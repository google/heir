# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@io_bazel_rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "binops",
    heir_opt_flags = [
        "--secretize",
        "--mlir-to-secret-arithmetic",
        "--secret-insert-mgmt-bgv",
        "--secret-distribute-generic",
        "--secret-to-bgv=poly-mod-degree=4",
        "--bgv-to-lattigo",
        "--cse",
    ],
    mlir_src = "binops.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.
go_test(
    name = "binops_test",
    srcs = ["binops_test.go"],
    embed = [":binops"],
    deps = [
        "@lattigo//core/rlwe",
        "@lattigo//schemes/bgv",
    ],
)
