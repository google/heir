# Tests depending on Yosys

load("@rules_python//python:py_binary.bzl", "py_binary")
load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

py_binary(
    name = "run_verilog",
    srcs = [
        "run_verilog.py",
    ],
    data = [
        "@at_clifford_yosys//:yosys",
    ],
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
    test_file_exts = [
        "mlir",
        "v",
    ],
)
