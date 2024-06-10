"""Macros for defining lit tests."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_python//python:py_test.bzl", "py_test")

_DEFAULT_FILE_EXTS = ["mlir"]

def lit_test(name = None, src = None, size = "small", tags = None, data = None):
    """Define a lit test.

    In its simplest form, a manually defined lit test would look like this:

      py_test(
          name = "ops.mlir.test",
          srcs = ["@llvm_project//llvm:lit"],
          args = ["-v", "tests/ops.mlir"],
          data = [":test_utilities", ":ops.mlir"],
          size = "small",
          main = "lit.py",
      )

    Where the `ops.mlir` file contains the test cases in standard RUN + CHECK format.

    The adjacent :test_utilities target contains all the tools (like mlir-opt) and
    files (like lit.cfg.py) that are needed to run a lit test. lit.cfg.py further
    specifies the lit configuration, including augmenting $PATH to include heir-opt.

    This macro simplifies the above definition by filling in the boilerplate.

    Args:
      name: the name of the test.
      src: the source file for the test.
      size: the size of the test.
      tags: tags to pass to the target.
      data: the data to pass to the target.
    """
    if not src:
        fail("src must be specified")
    name = name or src + ".test"

    filegroup_name = name + ".filegroup"
    native.filegroup(
        name = filegroup_name,
        srcs = [src],
    )

    if not data:
        data = []
    data = data + [filegroup_name]

    py_test(
        name = name,
        size = size,
        # -v ensures lit outputs useful info during test failures
        args = ["-v", paths.join(native.package_name(), src)],
        data = data,
        srcs = ["@llvm-project//llvm:lit"],
        main = "lit.py",
        tags = tags,
    )

def glob_lit_tests(
        # these unused args are kept for API compatibility with the corresponding
        # google-internal macro
        name = None,  # buildifier: disable=unused-variable
        data = None,
        driver = None,  # buildifier: disable=unused-variable
        exclude = None,
        size_override = None,
        test_file_exts = None,
        default_tags = None,
        tags_override = None):
    """Searches the caller's directory for files to run as lit tests.

    Args:
      name: unused
      data: the data to pass to the test, e.g., targets containing binaries
        used by the lit test in a RUN command.
      driver: unused
      exclude: a list of glob patterns to exclude
      size_override: a dictionary giving per-source-file test size overrides
      test_file_exts: a list of file extensions to use for globbing for files
        that should be defined as tests.
      default_tags: [str] tags to add to each test
      tags_override: tags to pass to each generated target.
    """
    exclude = exclude or []
    test_file_exts = test_file_exts or _DEFAULT_FILE_EXTS
    size_override = size_override or dict()
    tags_override = tags_override or dict()
    default_tags = default_tags or []
    tests = native.glob(["*." + ext for ext in test_file_exts], exclude = exclude)

    for curr_test in tests:
        lit_test(
            src = curr_test,
            size = size_override.get(curr_test, "small"),
            tags = default_tags + tags_override.get(curr_test, []),
            data = data,
        )
