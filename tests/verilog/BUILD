# Tests depending on Yosys

load("@rules_python//python:py_binary.bzl", "py_binary")
load("//bazel:lit.bzl", "glob_lit_tests")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

py_binary(
    name = "run_verilog",
    srcs = [
        "run_verilog.py",
    ],
    data = [
        "@at_clifford_yosys//:yosys",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
)

glob_lit_tests(
    name = "all_tests",
    data = [
        ":run_verilog",
        "@at_clifford_yosys//:yosys",
        "@heir//tests:test_utilities",
    ],
    default_tags = ["yosys"],
    driver = "@heir//tests:run_lit.sh",
    size_override = {
        "hello_world.tosa.mlir": "large",
    },
    tags_override = {
        "hello_world.tosa.mlir": ["nofastbuild"],
    },
    test_file_exts = [
        "mlir",
        "v",
    ],
)
