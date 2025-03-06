load("//tests/Examples/plaintext:test.bzl", "plaintext_test")

package(default_applicable_licenses = ["@heir//:license"])

plaintext_test(
    src = "dot_product_8f.mlir",
    data = [
        ":dot_product_8f_test",
    ],
)

cc_library(
    name = "dot_product_8f_test",
    srcs = ["dot_product_8f_test.c"],
)
