load("@heir//tests/Examples/plaintext:test.bzl", "plaintext_test")

package(default_applicable_licenses = ["@heir//:license"])

cc_library(
    name = "add_test",
    srcs = ["add_test.c"],
)

# Secret arithmetic plaintext backend with original integer type semantic
plaintext_test(
    name = "add",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend",
    ],
    mlir_src = "add.mlir",
    deps = [
        ":add_test",
        "@heir//tests/Examples/plaintext:memrefCopy",
    ],
)
