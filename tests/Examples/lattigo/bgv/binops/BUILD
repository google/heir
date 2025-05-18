# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "binops",
    go_library_name = "binops",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=bgv",
        "--mlir-to-bgv=ciphertext-degree=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "binops.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.

go_test(
    name = "binops_test",
    srcs = ["binops_test.go"],
    embed = [":binops"],
)
