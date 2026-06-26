load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "batchnorm_sigmoid_test",
    generated_lib_header = "batchnorm_sigmoid_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--activation-canonicalizations",
        "--mlir-to-ckks=ciphertext-degree=1024",
        "--scheme-to-openfhe",
    ],
    heir_translate_flags = [],
    mlir_src = "batchnorm_sigmoid.mlir",
    tags = ["nofastbuild"],
    test_src = "batchnorm_sigmoid_test.cpp",
)
