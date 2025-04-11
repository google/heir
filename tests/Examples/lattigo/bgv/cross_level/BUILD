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
        "--annotate-module=backend=lattigo scheme=bgv",
        "--mlir-to-bgv=ciphertext-degree=1024",
        "--scheme-to-lattigo=insert-debug-handler-calls=true",
    ],
    mlir_src = "cross_level.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.

go_test(
    name = "crossleveldebug_test",
    srcs = ["cross_level_debug_test.go"],
    embed = [":main"],
)
