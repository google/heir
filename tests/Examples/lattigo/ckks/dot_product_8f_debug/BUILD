# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# This is a Google-internal hack to avoid issues with automated tooling that
# expects very specific go package layout with respect to build targets.
# @unused
glaze_ignore = [
    "dot_product_8f_debug.go",
]

heir_lattigo_lib(
    name = "dot_product_8f_debug",
    extra_srcs = ["dot_product_8f_debug.go"],
    go_library_name = "dotproduct8fdebug",
    heir_opt_flags = [
        "--mlir-to-ckks=ciphertext-degree=8",
        "--scheme-to-lattigo=insert-debug-handler-calls=true",
    ],
    mlir_src = "dot_product_8f.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.

go_test(
    name = "dotproduct8fdebug_test",
    srcs = ["dot_product_8f_debug_test.go"],
    embed = [":dotproduct8fdebug"],
)
