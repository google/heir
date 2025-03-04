"""A macro providing an end-to-end test for Plaintext Backend."""

load("@heir//bazel:lit.bzl", "lit_test")

def plaintext_test(src, size = "small", data = [], tags = []):
    """Define a lit test for the Plaintext Backend.

    Args:
      src: The source mlir file to run through lit.
      size: The size of the test.
      data: Data dependencies to be passed to lit_test
      tags: Tags to pass to lit_test
    """

    _data = data + [
        "@heir//tests:test_utilities",
        "@llvm-project//llvm:llc",
        "@llvm-project//mlir:mlir-translate",
    ]

    _tags = tags + [
        "notap",
    ]

    lit_test(
        src = src,
        size = size,
        data = _data,
        tags = _tags,
    )
