load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test", "openfhe_interpreter_test")

package(default_applicable_licenses = ["@heir//:license"])

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
    tags = ["notap"],
    test_src = "halevi_shoup_matvec_test.cpp",
)

openfhe_interpreter_test(
    name = "halevi_shoup_matvec_interpreter_test",
    generated_heir_opt_filename = "module.openfhe.mlir",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=8192",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/common:matvec.mlir",
    test_src = "halevi_shoup_matvec_interpreter_test.cpp",
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)
