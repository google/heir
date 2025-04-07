load("@heir//tests/Examples/plaintext:test.bzl", "plaintext_test")

package(default_applicable_licenses = ["@heir//:license"])

cc_library(
    name = "dot_product_8f_debug_test",
    srcs = ["dot_product_8f_debug_test.c"],
)

plaintext_test(
    name = "dot_product_8f_debug",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=insert-debug-handler-calls=true",
    ],
    # log is visible to crypto backend
    log_file_name = "dot_product_8f_debug.log",
    log_file_visibility = ["@heir//tests/Examples:__subpackages__"],
    mlir_src = "@heir//tests/Examples/common:dot_product_8f.mlir",
    deps = [
        ":dot_product_8f_debug_test",
        "@heir//tests/Examples/plaintext:memrefCopy",
    ],
)
