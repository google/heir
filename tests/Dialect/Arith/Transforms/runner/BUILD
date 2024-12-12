load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = [
        "@//tests/Dialect/Arith/Transforms/runner:quarter_to_llvm.c",
        "@heir//tests:test_utilities",
        "@llvm-project//clang:clang",
        "@llvm-project//llvm:llc",
        "@llvm-project//mlir:mlir-translate",
    ],
    default_tags = [
        "manual",
        "notap",
    ],
    driver = "@heir//tests:run_lit.sh",
    test_file_exts = ["mlir"],
)
