load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

py_binary(
    name = "generate_static_roots",
    srcs = ["generate_static_roots.py"],
    deps = ["@heir_pip_deps//sympy"],
)

py_test(
    name = "test_lit_to_bazel",
    srcs = ["test_lit_to_bazel.py"],
    deps = [
        ":lit_to_bazel_lib",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_test(
    name = "test_e2e_to_bazel",
    srcs = ["test_e2e_to_bazel.py"],
    deps = [
        ":e2e_to_bazel_lib",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_library(
    name = "lit_to_bazel_lib",
    srcs = ["lit_to_bazel_lib.py"],
    deps = [
        "@heir_pip_deps//fire",
    ],
)

py_binary(
    name = "lit_to_bazel",
    srcs = ["lit_to_bazel.py"],
    deps = [
        ":lit_to_bazel_lib",
        "@heir_pip_deps//fire",
    ],
)

py_library(
    name = "e2e_to_bazel_lib",
    srcs = ["e2e_to_bazel_lib.py"],
    deps = [
        "@heir_pip_deps//fire",
    ],
)

py_binary(
    name = "e2e_to_bazel",
    srcs = ["e2e_to_bazel.py"],
    deps = [
        ":e2e_to_bazel_lib",
        "@heir_pip_deps//fire",
    ],
)

py_library(
    name = "get_version",
    srcs = ["get_version.py"],
    deps = [
        "@heir_pip_deps//requests",
    ],
)

py_test(
    name = "test_get_version",
    srcs = ["test_get_version.py"],
    deps = [
        ":get_version",
        "@abseil-py//absl/testing:absltest",
    ],
)
