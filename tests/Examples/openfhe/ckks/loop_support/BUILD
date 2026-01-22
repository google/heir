load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "loop_test",
    generated_lib_header = "loop_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=8 scaling-mod-bits=55 first-mod-bits=60 level-budget=3 modulus-switch-after-mul=true experimental-disable-loop-unroll=true",
        "--scheme-to-openfhe=insert-debug-handler-calls=true",
    ],
    heir_translate_flags = [
        "--openfhe-debug-helper-include-path=tests/Examples/openfhe/ckks/loop_support/debug_helper.h",
    ],
    mlir_src = "@heir//tests/Examples/common:loop.mlir",
    test_src = "loop_test.cpp",
    deps = [
        "@heir//tests/Examples/openfhe/ckks/loop_support:debug_helper",
    ],
)

cc_library(
    name = "debug_helper",
    srcs = ["debug_helper.cpp"],
    hdrs = ["debug_helper.h"],
    deps = [
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)
