load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "jaxite_lib",
    srcs = glob(
        ["**/*.py"],
        exclude = [
            "**/*_test.py",
            "**/test_util.py",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        "@jaxite_deps_jax//:pkg",
        "@jaxite_deps_jaxlib//:pkg",
        # copybara: jax:pallas_lib
        # copybara: jax:pallas_tpu
    ],
)