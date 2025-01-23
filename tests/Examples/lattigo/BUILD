# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@io_bazel_rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "binops",
    heir_opt_flags = [
        "--mlir-to-lattigo-bgv=entry-function=add ciphertext-degree=4",
    ],
    mlir_src = "binops.mlir",
)

heir_lattigo_lib(
    name = "dotproduct8",
    heir_opt_flags = [
        "--mlir-to-lattigo-bgv=entry-function=dot_product ciphertext-degree=8",
    ],
    mlir_src = "dot_product_8.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.
go_test(
    name = "binops_test",
    srcs = ["binops_test.go"],
    embed = [":binops"],
)

go_test(
    name = "dotproduct8_test",
    srcs = ["dot_product_8_test.go"],
    embed = [":dotproduct8"],
)
