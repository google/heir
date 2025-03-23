load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    # TODO(#1181): make level analysis work for affine loop
    # TODO(#1569): make scale analysis work
    # TODO(#1606): fix distribute generic over loop
    exclude = [
        "matmul_loop.mlir",
        "loop.mlir",
        "matmul_unrolled.mlir",
    ],
    test_file_exts = ["mlir"],
)
