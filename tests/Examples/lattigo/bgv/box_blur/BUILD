# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "box_blur",
    go_library_name = "boxblur",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=bgv",
        "--mlir-to-bgv=plaintext-modulus=786433 ciphertext-degree=2048",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:box_blur_16x16.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.

go_test(
    name = "boxblur_test",
    srcs = ["box_blur_test.go"],
    embed = [":boxblur"],
)
