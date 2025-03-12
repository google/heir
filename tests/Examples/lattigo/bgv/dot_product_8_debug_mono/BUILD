# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# This is a Google-internal hack to avoid issues with automated tooling that
# expects very specific go package layout with respect to build targets.
# @unused
glaze_ignore = [
    "dot_product_8_debug.go",
]

heir_lattigo_lib(
    name = "dot_product_8_debug",
    extra_srcs = ["dot_product_8_debug.go"],
    go_library_name = "dotproduct8debug",
    heir_opt_flags = [
        "--mlir-to-bgv=ciphertext-degree=8 noise-model=bgv-noise-mono-pk annotate-noise-bound=true",
        "--scheme-to-lattigo=insert-debug-handler-calls=true",
    ],
    mlir_src = "dot_product_8.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.

go_test(
    name = "dotproduct8debug_test",
    srcs = ["dot_product_8_debug_handler_test.go"],
    embed = [":dotproduct8debug"],
)
