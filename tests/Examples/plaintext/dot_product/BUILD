load("//tests/Examples/plaintext:test.bzl", "plaintext_test")

package(default_applicable_licenses = ["@heir//:license"])

plaintext_test(
    src = "dot_product_8.mlir",
    data = [
        ":dot_product_8_mod_test",
        ":dot_product_8_test",
    ],
)

cc_library(
    name = "dot_product_8_test",
    srcs = ["dot_product_8_test.c"],
)

cc_library(
    name = "dot_product_8_mod_test",
    srcs = ["dot_product_8_mod_test.c"],
)
