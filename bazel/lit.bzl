"""A simple macro to define a lit test."""

_TESTS_DIR = "tests"

def lit_test(name=None, src=None, size="small"):
  """Define a lit test.

  In its simplest form, a manually defined lit test would look like this:

    py_test(
        name = "ops.mlir.test",
        srcs = ["@llvm-project//llvm:lit"],
        args = ["-v", "tests/ops.mlir"],
        data = [":test_utilities", ":ops.mlir"],
        size = "small",
        main = "lit.py",
    )

  Where the `ops.mlir` file contains the test cases in standard RUN + CHECK format.

  The adjacent :test_utilities target contains all the tools (like mlir-opt) and
  files (like lit.cfg.py) that are needed to run a lit test. lit.cfg.py further
  specifies the lit configuration, including augmenting the path to heir-opt.

  This macro simplifies the definition by filling in the boilerplate.
  """
  if not src:
    fail("src must be specified")
  name = name or src + ".test"
  rel_target = ":" + src

  native.py_test(
      name = name,
      size = size,
      # -v ensures lit outputs useful info during test failures
      args = ["-v", _TESTS_DIR + "/" + src],
      data = ["@heir//tests:test_utilities", rel_target],
      srcs = ["@llvm-project//llvm:lit"],
      main = "lit.py",
  )
