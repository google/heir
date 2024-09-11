load("@heir//tests/jaxite/end_to_end:test.bzl", "jaxite_end_to_end_test")
load("@rules_python//python:py_library.bzl", "py_library")
load("//bazel:lit.bzl", "glob_lit_tests")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    test_file_exts = ["mlir"],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = ["@heir_pip_deps_jaxite//:pkg"],
)

jaxite_end_to_end_test(
    name = "add_one_lut3",
    heir_opt_flags = "--tosa-to-boolean-jaxite=entry-function=test_add_one_lut3",
    mlir_src = "add_one_lut3.mlir",
    test_src = "add_one_lut3_test.py",
    deps = [":test_utils"],
)
