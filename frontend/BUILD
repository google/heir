load("@rules_python//python:py_library.bzl", "py_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

DATA_DEPS = [
    "@heir//tools:heir-opt",
    "@heir//tools:heir-translate",
    # copybara: python_runtime_headers
]

# a single-source build dependency that gives the whole (non-test) source tree;
# note we chose the style of putting all test rules below, because glob does
# not recurse into subdirectories with BUILD files in them.
py_library(
    name = "frontend",
    srcs = glob(
        ["heir/**/*.py"],
        exclude = [
            "**/*_test.py",
        ],
    ),
    data = DATA_DEPS,
    deps = [
        "@heir_pip_deps//colorama",
        "@heir_pip_deps//numba",
        "@heir_pip_deps//numpy",
        "@heir_pip_deps//pybind11",
        "@heir_pip_deps//pybind11_global",
    ],
)

# Because tests require JIT-compiling generated code on the fly, and linking
# with system-provided libs, we cannot run them inside bazel's hermetic C++
# toolchain. Instead the tests are run as a separate CI test, after installing
# the frontend package, using pytest to discover and run tests named
# frontend/**/*_test.py
