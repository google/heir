load("@heir//tests/Examples/plaintext:test.bzl", "plaintext_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_applicable_licenses = ["@heir//:license"])

cc_library(
    name = "roberts_cross_test",
    srcs = [
        "roberts_cross_test.c",
    ],
)

cc_library(
    name = "roberts_cross_mod_test",
    srcs = [
        "roberts_cross_mod_test.c",
    ],
)

cc_library(
    name = "roberts_cross_mod_failure_test",
    srcs = [
        "roberts_cross_mod_test.c",
    ],
    defines = [
        "EXPECT_FAILURE",
    ],
)

# Secret arithmetic plaintext backend with original integer type semantic
plaintext_test(
    name = "roberts_cross",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=4096",
    ],
    mlir_src = "@heir//tests/Examples/common:roberts_cross_16x16.mlir",
    deps = [
        ":roberts_cross_test",
        "@heir//tests/Examples/plaintext:memrefCopy",
    ],
)

# Secret arithmetic plaintext backend with plaintext modulus semantic
plaintext_test(
    name = "roberts_cross_mod_536903681",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=4096 plaintext-modulus=536903681",
    ],
    mlir_src = "@heir//tests/Examples/common:roberts_cross_16x16.mlir",
    deps = [
        ":roberts_cross_mod_test",
        "@heir//tests/Examples/plaintext:memrefCopy",
    ],
)

# Secret arithmetic plaintext backend with plaintext modulus semantic
plaintext_test(
    name = "roberts_cross_mod_786433",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=4096 plaintext-modulus=786433",
    ],
    mlir_src = "@heir//tests/Examples/common:roberts_cross_16x16.mlir",
    deps = [
        ":roberts_cross_mod_failure_test",
        "@heir//tests/Examples/plaintext:memrefCopy",
    ],
)
