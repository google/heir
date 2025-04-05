load("@heir//tests/Examples/plaintext:test.bzl", "plaintext_test")

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
    data = ["@heir//tests/Examples/common"],
    heir_opt_flags = [
        "--mlir-to-plaintext-backend",
    ],
    mlir_src = "@heir//tests/Examples/common:roberts_cross_64x64.mlir",
    deps = [
        ":roberts_cross_test",
        "@heir//tests/Examples/plaintext:memrefCopy",
    ],
)

# Secret arithmetic plaintext backend with plaintext modulus semantic
plaintext_test(
    name = "roberts_cross_mod_536903681",
    data = ["@heir//tests/Examples/common"],
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-modulus=536903681",
    ],
    mlir_src = "@heir//tests/Examples/common:roberts_cross_64x64.mlir",
    deps = [
        ":roberts_cross_mod_test",
        "@heir//tests/Examples/plaintext:memrefCopy",
    ],
)

# Secret arithmetic plaintext backend with plaintext modulus semantic
plaintext_test(
    name = "roberts_cross_mod_786433",
    data = ["@heir//tests/Examples/common"],
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-modulus=786433",
    ],
    mlir_src = "@heir//tests/Examples/common:roberts_cross_64x64.mlir",
    deps = [
        ":roberts_cross_mod_failure_test",
        "@heir//tests/Examples/plaintext:memrefCopy",
    ],
)
