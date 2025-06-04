# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

go_test(
    name = "simplesum_test",
    srcs = ["simple_sum_test.go"],
    embed = [":simplesum"],
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.

heir_lattigo_lib(
    name = "simple_sum",
    go_library_name = "simplesum",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=bgv",
        "--mlir-to-bgv=ciphertext-degree=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:simple_sum.mlir",
)
