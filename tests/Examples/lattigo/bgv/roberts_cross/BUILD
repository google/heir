# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "roberts_cross",
    go_library_name = "robertscross",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=bgv",
        "--mlir-to-bgv=ciphertext-degree=4096 plaintext-modulus=536903681",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:roberts_cross_16x16.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.

go_test(
    name = "robertscross_test",
    srcs = ["roberts_cross_test.go"],
    embed = [":robertscross"],
)
