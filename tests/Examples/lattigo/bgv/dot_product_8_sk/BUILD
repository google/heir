# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "dot_product_8_sk",
    go_library_name = "dotproduct8sk",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=bgv",
        "--mlir-to-bgv=ciphertext-degree=1024 use-public-key=false",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:dot_product_8.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.

go_test(
    name = "dotproduct8sk_test",
    srcs = ["dot_product_8_sk_test.go"],
    embed = [":dotproduct8sk"],
)
