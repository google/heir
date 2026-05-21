load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "entrypoint_lib",
    srcs = ["entrypoint.py"],
)

py_test(
    name = "test_entrypoint",
    srcs = ["test_entrypoint.py"],
    deps = [
        ":entrypoint_lib",
    ],
)
