load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "batch_matmul_test",
    generated_lib_header = "batch_matmul_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=min-slot-count=4096 experimental-disable-loop-unroll=true greedy-level-budget=10 first-mod-bits=60 scaling-mod-bits=50",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/common:batch_matmul.mlir",
    tags = ["nofastbuild"],
    test_src = "batch_matmul_test.cpp",
)
