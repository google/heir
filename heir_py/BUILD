load("@rules_python//python:defs.bzl", "py_library")
load("//heir_py:testing.bzl", "heir_py_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:private"],
)

# a single-source build dependency that gives the whole (non-test) source tree;
# note we chose the style of putting all test rules below, because glob does
# not recurse into subdirectories with BUILD files in them.
py_library(
    name = "heir_py",
    srcs = glob(
        ["**/*.py"],
        exclude = [
            "**/*_test.py",
        ],
    ),
    data = [
        "@heir//tools:heir-opt",
        "@heir//tools:heir-translate",
    ],
    deps = [
        "@heir_py_pip_deps_numba//:pkg",
        "@heir_py_pip_deps_pybind11//:pkg",
        "@heir_py_pip_deps_pybind11_global//:pkg",
    ],
)

heir_py_test(
    name = "pipeline_test",
    srcs = ["pipeline_test.py"],
)
