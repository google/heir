# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# This is a Google-internal hack to avoid issues with automated tooling that
# expects very specific go package layout with respect to build targets.
# @unused
glaze_ignore = [
    "cross_level_debug.go",
]

heir_lattigo_lib(
    name = "cross_level_debug",
    extra_srcs = ["cross_level_debug.go"],
    go_library_name = "main",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=4 modulus-switch-before-first-mul=true first-mod-bits=59 scaling-mod-bits=45",
        "--scheme-to-lattigo=insert-debug-handler-calls=true",
    ],
    mlir_src = "cross_level.mlir",
    tags = ["e2e"],
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.

go_test(
    name = "crossleveldebug_test",
    srcs = ["cross_level_debug_test.go"],
    embed = [":main"],
    tags = ["e2e"],
)
