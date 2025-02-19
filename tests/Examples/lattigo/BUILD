# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@io_bazel_rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# This is a Google-internal hack to avoid issues with automated tooling that
# expects very specific go package layout with respect to build targets.
# @unused
glaze_ignore = [
    "dot_product_8_debug.go",
]

# BGV

heir_lattigo_lib(
    name = "binops",
    go_library_name = "binops",
    heir_opt_flags = [
        "--mlir-to-bgv=ciphertext-degree=4",
        "--scheme-to-lattigo",
    ],
    mlir_src = "binops.mlir",
)

heir_lattigo_lib(
    name = "dot_product_8",
    go_library_name = "dotproduct8",
    heir_opt_flags = [
        "--mlir-to-bgv=ciphertext-degree=8",
        "--scheme-to-lattigo",
    ],
    mlir_src = "dot_product_8.mlir",
)

heir_lattigo_lib(
    name = "dot_product_8_sk",
    go_library_name = "dotproduct8sk",
    heir_opt_flags = [
        "--mlir-to-bgv=ciphertext-degree=8 use-public-key=false",
        "--scheme-to-lattigo",
    ],
    mlir_src = "dot_product_8.mlir",
)

heir_lattigo_lib(
    name = "dot_product_8_debug",
    extra_srcs = ["dot_product_8_debug.go"],
    go_library_name = "dotproduct8debug",
    heir_opt_flags = [
        "--mlir-to-bgv=ciphertext-degree=8",
        "--scheme-to-lattigo=insert-debug-handler-calls=true",
    ],
    mlir_src = "dot_product_8.mlir",
)

# CKKS

heir_lattigo_lib(
    name = "dot_product_8f",
    go_library_name = "dotproduct8f",
    heir_opt_flags = [
        "--mlir-to-ckks=ciphertext-degree=8",
        "--scheme-to-lattigo",
    ],
    mlir_src = "dot_product_8f.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.

# BGV

go_test(
    name = "binops_test",
    srcs = ["binops_test.go"],
    embed = [":binops"],
)

go_test(
    name = "dotproduct8_test",
    srcs = ["dot_product_8_test.go"],
    embed = [":dotproduct8"],
)

go_test(
    name = "dotproduct8sk_test",
    srcs = ["dot_product_8_sk_test.go"],
    embed = [":dotproduct8sk"],
)

go_test(
    name = "dotproduct8debug_test",
    srcs = ["dot_product_8_debug_handler_test.go"],
    embed = [":dotproduct8debug"],
)

# CKKS

go_test(
    name = "dotproduct8f_test",
    srcs = ["dot_product_8f_test.go"],
    embed = [":dotproduct8f"],
)
