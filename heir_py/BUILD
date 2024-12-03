load("@rules_python//python:defs.bzl", "py_library", "py_test")

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
        # FIXME: add pybind deps
        # "@heir_py_deps_pybind11//:pkg",
        # "@heir_py_deps_pybind11_global//:pkg",
    ],
)

# Test rules are below, though the source files are in subdirectories.
py_test(
    name = "pipeline_test",
    srcs = ["pipeline_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":heir_py",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)
