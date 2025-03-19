# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "halevi_shoup_matvec_test",
    data = ["@heir//tests/Examples/common"],
    generated_lib_header = "halevi_shoup_matvec_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=16",
        "--scheme-to-openfhe=insert-debug-handler-calls=true",
    ],
    heir_translate_flags = [
        "--openfhe-include-type=source-relative",
    ],
    mlir_src = "@heir//tests/Examples/common:halevi_shoup_matvec.mlir",
    tags = ["notap"],
    test_src = "halevi_shoup_matvec_test.cpp",
)
