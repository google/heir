# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "cmux",
    go_library_name = "main",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=bgv",
        "--mlir-to-bgv",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:cmux.mlir",
    tags = ["e2e"],
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.

go_test(
    name = "cmux_test",
    srcs = ["cmux_test.go"],
    embed = [":main"],
    tags = ["e2e"],
)
