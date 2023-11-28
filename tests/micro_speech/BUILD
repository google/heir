load("//bazel:lit.bzl", "glob_lit_tests")

package(
    default_applicable_licenses = ["@heir//:license"],
)

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    size_override = {
        "micro_speech.tosa.mlir": "large",
        "before_unroll_and_forward.mlir": "large",
    },
    tags_override = {
        "micro_speech.tosa.mlir": [
            "nofastbuild",
            "notap",
            "manual",
        ],
        "before_unroll_and_forward.mlir": [
            "nofastbuild",
            "manual",
        ],
    },
    test_file_exts = ["mlir"],
)
