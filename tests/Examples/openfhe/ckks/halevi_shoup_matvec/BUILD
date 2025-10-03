# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_applicable_licenses = ["@heir//:license"])

cc_library(
    name = "debug_helper",
    srcs = ["debug_helper.cpp"],
    hdrs = ["debug_helper.h"],
    deps = [
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)

# TODO(#2047): change to --mlir-to-ckks when we integrate the new layout system
openfhe_end_to_end_test(
    name = "halevi_shoup_matvec_test",
    generated_lib_header = "halevi_shoup_matvec_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=8192",
        "--scheme-to-openfhe=insert-debug-handler-calls=true",
    ],
    heir_translate_flags = [
        "--openfhe-include-type=source-relative",
    ],
    mlir_src = "@heir//tests/Examples/common:matvec.mlir",
    test_src = "halevi_shoup_matvec_test.cpp",
    deps = [":debug_helper"],
)
