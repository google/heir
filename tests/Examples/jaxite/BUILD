load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@heir//tests/Examples/jaxite:test.bzl", "jaxite_end_to_end_test")
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
    exclude = ["fully_connected.jaxite.mlir"],
    test_file_exts = ["mlir"],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    strict_deps = False,
    deps = ["@heir_pip_deps//jaxite"],
)

jaxite_end_to_end_test(
    name = "add_one_lut3",
    heir_opt_pass_flags = [
        "--mlir-to-cggi",
        "--scheme-to-jaxite",
    ],
    mlir_src = "add_one_lut3.mlir",
    tags = ["yosys"],
    test_src = "add_one_lut3_test.py",
    deps = [":test_utils"],
)

jaxite_end_to_end_test(
    name = "fully_connected",
    mlir_src = "fully_connected.jaxite.mlir",
    test_src = "fully_connected_test.py",
    deps = [":test_utils"],
)

exports_files([
    "add_one_lut3.mlir",
])

bzl_library(
    name = "test_bzl",
    srcs = ["test.bzl"],
    visibility = ["//visibility:public"],
    deps = [
        "@heir//tools:heir_jaxite_bzl",
        "@rules_python//python:py_test_bzl",
    ],
)
