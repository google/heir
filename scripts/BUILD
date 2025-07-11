load("@rules_python//python:py_binary.bzl", "py_binary")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

py_binary(
    name = "generate_static_roots",
    srcs = ["generate_static_roots.py"],
    deps = ["@heir_pip_deps//sympy"],
)
