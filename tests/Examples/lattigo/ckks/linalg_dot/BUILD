# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "linalg_dot",
    go_library_name = "linalgdot",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=2048 first-mod-bits=0",
        "--scheme-to-lattigo",
    ],
    mlir_src = "linalg_dot.mlir",
)

go_test(
    name = "linalgdot_test",
    srcs = ["linalg_dot_test.go"],
    embed = [":linalgdot"],
)
