load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")
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
llvm_runner_test(
    name = "roberts_cross",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=4096",
    ],
    mlir_src = "@heir//tests/Examples/common:roberts_cross_16x16.mlir",
    deps = [":roberts_cross_test"],
)

# Secret arithmetic plaintext backend with plaintext modulus semantic
llvm_runner_test(
    name = "roberts_cross_mod_786433",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=4096 plaintext-modulus=786433",
    ],
    mlir_src = "@heir//tests/Examples/common:roberts_cross_16x16.mlir",
    deps = [":roberts_cross_mod_test"],
)

# Secret arithmetic plaintext backend with plaintext modulus semantic
llvm_runner_test(
    name = "roberts_cross_mod_65537",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=4096 plaintext-modulus=65537",
    ],
    mlir_src = "@heir//tests/Examples/common:roberts_cross_16x16.mlir",
    deps = [":roberts_cross_mod_failure_test"],
)
