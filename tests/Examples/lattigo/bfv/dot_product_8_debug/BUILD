# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# This is a Google-internal hack to avoid issues with automated tooling that
# expects very specific go package layout with respect to build targets.
# @unused
glaze_ignore = [
    "dot_product_8_bfv_debug.go",
]

heir_lattigo_lib(
    name = "dot_product_8_bfv_debug",
    extra_srcs = ["dot_product_8_bfv_debug.go"],
    go_library_name = "dotproduct8bfvdebug",
    heir_opt_flags = [
        "--mlir-to-bfv=ciphertext-degree=8 annotate-noise-bound=true encryption-technique-extended=true",
        "--scheme-to-lattigo=insert-debug-handler-calls=true",
    ],
    mlir_src = "dot_product_8.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.

go_test(
    name = "dotproduct8bfvdebug_test",
    srcs = ["dot_product_8_bfv_debug_handler_test.go"],
    embed = [":dotproduct8bfvdebug"],
)
