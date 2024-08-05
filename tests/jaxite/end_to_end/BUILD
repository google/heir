load("@heir//tests/jaxite/end_to_end:test.bzl", "jaxite_end_to_end_test")
load("@rules_python//python:py_library.bzl", "py_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = ["@heir_pip_deps_jaxite//:pkg"],
)

jaxite_end_to_end_test(
    name = "add_one_lut3",
    heir_opt_flags = "--cggi-to-jaxite",
    mlir_src = "add_one_lut3.mlir",
    test_src = "add_one_lut3_test.py",
    deps = [":test_utils"],
)
