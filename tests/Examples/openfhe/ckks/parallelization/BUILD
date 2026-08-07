load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "parallelization_test",
    generated_lib_header = "parallelization_lib.h",
    heir_opt_flags = [
        "--openfhe-fast-rotation-precompute",
        "--boolean-vectorize",
        "--canonicalize",
        "--cse",
        "--openfhe-alloc-to-inplace",
    ],
    heir_translate_flags = [
        "--openfhe-include-type=source-relative",
    ],
    mlir_src = "parallelization.mlir",
    tags = ["notap"],
    test_src = "parallelization_test.cpp",
)

openfhe_end_to_end_test(
    name = "parallelization_skip_test",
    generated_lib_header = "parallelization_skip_lib.h",
    heir_opt_flags = [
        "--openfhe-fast-rotation-precompute",
        "--canonicalize",
        "--cse",
        "--openfhe-alloc-to-inplace",
    ],
    heir_translate_flags = [
        "--openfhe-include-type=source-relative",
    ],
    mlir_src = "parallelization.mlir",
    tags = ["notap"],
    test_src = "parallelization_skip_test.cpp",
)
