load("@heir//tests/Examples/plaintext:test.bzl", "plaintext_test")

package(default_applicable_licenses = ["@heir//:license"])

cc_library(
    name = "dot_product_8f_test",
    srcs = ["dot_product_8f_test.c"],
)

plaintext_test(
    name = "dot_product_8f",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend",
    ],
    mlir_src = "@heir//tests/Examples/common:dot_product_8f.mlir",
    deps = [
        ":dot_product_8f_test",
        "@heir//tests/Examples/plaintext:memrefCopy",
    ],
)
