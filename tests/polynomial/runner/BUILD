load("@heir//tests/benchmark:benchmark.bzl", "heir_benchmark_test")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

py_binary(
    name = "generate_test_cases",
    srcs = [
        "generate_test_cases.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@heir_pip_deps_sympy//:pkg",
        "@heir_pip_deps_tomli//:pkg",
    ],
)

heir_benchmark_test(
    name = "ntt_benchmark_test",
    heir_opt_flags = "--heir-polynomial-to-llvm",
    mlir_src = "ntt_benchmark.mlir",
    test_src = ["ntt_benchmark_test.cc"],
    deps = [
        "@google_benchmark//:benchmark_main",
        "@googletest//:gtest",
        "@heir//tests/benchmark:Memref",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [
        "@heir//tests:test_utilities",
    ],
    driver = "@heir//tests:run_lit.sh",
    exclude = ["ntt_benchmark.mlir"],
    test_file_exts = ["mlir"],
)
