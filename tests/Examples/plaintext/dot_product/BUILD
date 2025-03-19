load("@heir//tests/Examples/plaintext:test.bzl", "plaintext_test")

package(default_applicable_licenses = ["@heir//:license"])

cc_library(
    name = "dot_product_8_test",
    srcs = ["dot_product_8_test.c"],
)

cc_library(
    name = "dot_product_8_mod_test",
    srcs = ["dot_product_8_mod_test.c"],
)

cc_library(
    name = "dot_product_8_mod_failure_test",
    srcs = ["dot_product_8_mod_test.c"],
    defines = [
        "EXPECT_FAILURE",
    ],
)

# Secret arithmetic plaintext backend with original integer type semantic
plaintext_test(
    name = "dot_product_8",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend",
    ],
    mlir_src = "@heir//tests/Examples/common:dot_product_8.mlir",
    deps = [
        ":dot_product_8_test",
        "@heir//tests/Examples/plaintext:memrefCopy",
    ],
)

# Secret arithmetic plaintext backend with plaintext modulus semantic
plaintext_test(
    name = "dot_product_8_mod_257",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-modulus=257",
    ],
    mlir_src = "@heir//tests/Examples/common:dot_product_8.mlir",
    deps = [
        ":dot_product_8_mod_test",
        "@heir//tests/Examples/plaintext:memrefCopy",
    ],
)

# Secret arithmetic plaintext backend with plaintext modulus semantic
plaintext_test(
    name = "dot_product_8_mod_65537",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-modulus=65537",
    ],
    mlir_src = "@heir//tests/Examples/common:dot_product_8.mlir",
    deps = [
        ":dot_product_8_mod_test",
        "@heir//tests/Examples/plaintext:memrefCopy",
    ],
)

# Secret arithmetic plaintext backend with plaintext modulus semantic
plaintext_test(
    name = "dot_product_8_mod_17",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-modulus=17",
    ],
    mlir_src = "@heir//tests/Examples/common:dot_product_8.mlir",
    deps = [
        ":dot_product_8_mod_failure_test",
        "@heir//tests/Examples/plaintext:memrefCopy",
    ],
)
