load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

filegroup(
    name = "orbit_cost_model",
    srcs = [
        "orbit_bad_cost_model.json",
        "orbit_cost_model.json",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [
        ":orbit_cost_model",
        "@heir//tests:test_utilities",
    ],
    driver = "@heir//tests:run_lit.sh",
    test_file_exts = ["mlir"],
)
