load("@heir//tests/Examples/benchmark:benchmark.bzl", "heir_benchmark_test")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

py_binary(
    name = "generate_test_cases",
    srcs = [
        "generate_test_cases.py",
    ],
    deps = [
        "@heir_pip_deps//sympy",
        "@heir_pip_deps//tomli",
    ],
)

# heir_benchmark_test(
#     name = "ntt_benchmark_test",
#     heir_opt_flags = ["--heir-polynomial-to-llvm"],
#     mlir_src = "ntt_benchmark.mlir",
#     test_src = ["ntt_benchmark_test.cc"],
#     deps = [
#         "@googletest//:gtest",
#         "@googletest//:gtest_main",
#         "@heir//tests/Examples/benchmark:Memref",
#     ],
# )

glob_lit_tests(
    name = "all_tests",
    data = [
        "@heir//tests:test_utilities",
    ],
    default_tags = ["notap"],
    driver = "@heir//tests:run_lit.sh",
    exclude = [
        "ntt_benchmark.mlir",
        # TODO(#1862): re-enable after replacing affine.for with scf.for
        "lower_ntt_perf_runner.mlir",
    ],
    test_file_exts = ["mlir"],
)
