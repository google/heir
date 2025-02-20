load("@heir//frontend:testing.bzl", "frontend_test")
load("@rules_python//python:py_library.bzl", "py_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

DATA_DEPS = [
    "@cereal//:headers",
    "@heir//tools:heir-opt",
    "@heir//tools:heir-translate",
    "@openfhe//:headers",
    "@openfhe//:binfhe",
    # copybara: openfhe_binfhe_headers
    "@openfhe//:core",
    # copybara: openfhe_core_headers
    "@openfhe//:pke",
    # copybara: openfhe_pke_headers
    # copybara: python_runtime_headers
    "@rapidjson//:headers",
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
        "@frontend_pip_deps_colorama//:pkg",
        "@frontend_pip_deps_numba//:pkg",
        "@frontend_pip_deps_pybind11//:pkg",
        "@frontend_pip_deps_pybind11_global//:pkg",
    ],
)

frontend_test(
    name = "e2e_test",
    srcs = ["e2e_test.py"],
    tags = [
        # copybara: manual
        "notap",
    ],
)

frontend_test(
    name = "loop_test",
    srcs = ["loop_test.py"],
    tags = [
        # copybara: manual
        "notap",
    ],
)
