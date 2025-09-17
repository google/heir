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
    "@openfhe//:libopenfhe",
    "@openfhe//:headers",
    # copybara: openfhe_binfhe_headers
    # copybara: openfhe_core_headers
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
        "@heir_pip_deps//colorama",
        "@heir_pip_deps//numba",
        "@heir_pip_deps//numpy",
        "@heir_pip_deps//pybind11",
        "@heir_pip_deps//pybind11_global",
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

frontend_test(
    name = "tensor_test",
    srcs = ["tensor_test.py"],
    tags = [
        # copybara: manual
        "notap",
    ],
)

frontend_test(
    name = "tensor_loop_test",
    srcs = ["tensor_loop_test.py"],
    tags = [
        # copybara: manual
        "notap",
    ],
)

frontend_test(
    name = "cggi_test",
    srcs = ["cggi_test.py"],
    tags = [
        # copybara: manual
        "notap",
    ],
)

frontend_test(
    name = "cast_test",
    srcs = ["cast_test.py"],
    tags = [
        # copybara: manual
        "notap",
    ],
)

frontend_test(
    name = "unique_name_test",
    srcs = ["unique_name_test.py"],
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

py_library(
    name = "example",
    srcs = ["example.py"],
    deps = [":frontend"],
)
