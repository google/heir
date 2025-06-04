# See README.md for setup required to run these tests

# load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")

package(default_applicable_licenses = ["@heir//:license"])

cc_library(
    name = "halevi_shoup_matmul_loop_test",
    testonly = True,
    srcs = ["halevi_shoup_matmul_loop_test.cpp"],
    deps = [
        "@googletest//:gtest_for_library_testonly",
        "@openfhe//:pke",
    ],
)

# cc_library(
#     name = "halevi_shoup_matmul_loop_test",
#     testonly = True,
#     srcs = ["halevi_shoup_matmul_loop_test.cpp"],
#     deps = [
#         "@openfhe//:pke",
#     ],
# )

# TODO(#1517): Re-enable this test once the parameters are fixed.
# openfhe_end_to_end_test(
#     name = "halevi_shoup_matmul_loop_test",
#     generated_lib_header = "halevi_shoup_matmul_loop_lib.h",
#     heir_opt_flags = [
#         "--annotate-module=backend=openfhe scheme=ckks",
#         "--mlir-to-ckks=ciphertext-degree=8192 experimental-disable-loop-unroll=true",
#         "--scheme-to-openfhe=entry-function=matmul",
#     ],
#     heir_translate_flags = [
#         "--openfhe-include-type=source-relative",
#     ],
#     mlir_src = "@heir//tests/Examples/common:halevi_shoup_matmul.mlir",
#
#     tags = ["notap"],
#     test_src = "halevi_shoup_matmul_loop_test.cpp",
# )
