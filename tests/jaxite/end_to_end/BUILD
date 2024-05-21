load("@heir//tests/jaxite/end_to_end:test.bzl", "jaxite_end_to_end_test")
load("@rules_python//python:py_library.bzl", "py_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = ["@jaxite//:jaxite_lib"],
)

jaxite_end_to_end_test(
    name = "add_one_lut3",
    mlir_src = "add_one_lut3.mlir",
    tags = ["notap"],
    test_src = "add_one_lut3_test.py",
    deps = [":test_utils"],
)
