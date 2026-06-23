load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "halevi_shoup_matvec_test",
    generated_lib_header = "halevi_shoup_matvec_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=8192",
        "--scheme-to-openfhe",
    ],
    heir_translate_flags = [
        "--openfhe-include-type=source-relative",
    ],
    mlir_src = "@heir//tests/Examples/common:matvec.mlir",
    tags = ["nofastbuild"],
    test_src = "halevi_shoup_matvec_test.cpp",
)
