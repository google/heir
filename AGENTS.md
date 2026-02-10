# AGENTS.md

## Bazel conventions

- Our build files are `BUILD` in almost all cases, not `BUILD.bazel`.

## MLIR conventions

- When creating operations, always use the format
  `OpType::create(rewriter, ...)` and never use the (deprecated) format
  `rewriter.create<OpType>(...)`.
- When writing unit tests with filecheck, never use `CHECK-LABEL`; just use
  `CHECK` instead.

## Helper tooling

### template scripts

- `scripts/templates/templates.py` (which requires `python-fire` and `jinja2`)
  provides boilerplate for generating new transforms and passes. After running
  the script to create a new pass, one still must register the pass in
  `heir-opt.cpp` and add the necessary build dependencies.

### heir-opt flags

Copied from https://mlir.llvm.org/docs/Tutorials/MlirOpt/ with minor tweaks.

- `--mlir-timing` displays execution times of each pass.
- `--debug` prints all debug information produced by `LLVM_DEBUG` or `LDBG`
  calls.
- `--debug-only="my-tag"` prints only the debug information produced by
  `LLVM_DEBUG` in files that have the macro `#define DEBUG_TYPE "my-tag"`. This
  often allows you to print only debug information associated with a specific
  pass.
  - `"greedy-rewriter"` only prints debug information for patterns applied with
    the greedy rewriter engine.
  - `"dialect-conversion"` only prints debug information for the dialect
    conversion framework.
- `--dump-pass-pipeline` prints the full pipeline that will be run before
  starting.
- `--mlir-print-ir-after-all` prints the IR after each pass.
  - See also `--mlir-print-ir-after-change`, `--mlir-print-ir-after-failure`,
    and analogous versions of these flags with `before` instead of `after`.
  - When using `print-ir` flags, adding `--mlir-print-ir-tree-dir` writes the
    IRs to files in a directory tree, making them easier to inspect versus a
    large dump to the terminal. A common practice when encountering a segfault
    is to create a minimal reproducing test case by adding
    `--dump-pass-pipeline --mlir-print-ir-before-all --mlir-print-ir-tree-dir=/tmp/mlir`,
    copying the last IR file (filenames are prefixed by the number of the pass
    in the pipeline) and running the single failing pass on it, using the output
    of `--dump-pass-pipeline` to get the pass options.

<!-- mdformat global-off -->
