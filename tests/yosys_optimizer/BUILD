load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

filegroup(
    name = "yosys_test_utilities",
    testonly = True,
    data = [
        "@edu_berkeley_abc//:abc",
        "@heir//lib/Transforms/YosysOptimizer/yosys:share_files",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [
        ":yosys_test_utilities",
        "@heir//tests:test_utilities",
    ],
    # A select on the test files and data for this test based on HEIR_NO_YOSYS
    # won't work because selects do not expand in macros.
    default_tags = ["yosys"],
    driver = "@heir//tests:run_lit.sh",
    size_override = {
        "micro_speech_for.mlir": "large",
        "64_bit_mul.mlir": "large",
    },
    tags_override = {
        "64_bit_mul.mlir": [
            "nofastbuild",
            "notap",
            "manual",
        ],
    },
    test_file_exts = ["mlir"],
)
