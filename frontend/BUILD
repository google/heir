load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
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
        "@frontend_pip_deps//colorama",
        "@frontend_pip_deps//numba",
        "@frontend_pip_deps//numpy",
        "@frontend_pip_deps//pybind11",
        "@frontend_pip_deps//pybind11_global",
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

frontend_test(
    name = "mixed_bitwidth_test",
    srcs = ["mixed_bitwidth_test.py"],
    tags = [
        # copybara: manual
        "notap",
    ],
)

bzl_library(
    name = "testing_bzl",
    srcs = ["testing.bzl"],
    visibility = ["//visibility:public"],
    deps = ["@rules_python//python:defs_bzl"],
)
