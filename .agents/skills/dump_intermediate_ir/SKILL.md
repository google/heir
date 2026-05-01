---
name: dump-intermediate-ir
description: >-
  Augments a `bazel run` CLI command for `heir-opt` with additional flags
  that dump the IR before each compiler pass, allowing the agent to inspect
  intermediate IR during compilation, identify a particular pass at which the
  pipeline fails, or extract the corresponding IR input to a particular pass
  for further debugging.
---

# Dump intermediate IR during compilation

## Overview

This skill guides the agent in adding flags to the `heir-opt` tool to dump the
compiled IR after each compiler pass to a series of files. These files can then
be inspected to identify the failing pass as well as the IR input to the failing
pass.

## Usage

### Converting a `bazel run` command

To convert a `bazel run` command, add the flags `--mlir-print-ir-before-all` and
`--mlir-print-ir-tree-dir=<PATH>`, where `<PATH>` is a path to a temporary
directory.

For example, given the following input command (which can be produced by the
`lit-to-bazel` skill),

```bash
bazel run //tools:heir-opt -- --mlir-to-ckks='ciphertext-degree=8' --scheme-to-openfhe='entry-function=dot_product' /path/to/file
```

The agent should produce the following (again where `<PATH>` is chosen
appropriately)

```bash
bazel run //tools:heir-opt -- --mlir-to-ckks='ciphertext-degree=8' --scheme-to-openfhe='entry-function=dot_product' --mlir-print-ir-before-all --mlir-print-ir-tree-dir=<PATH> /path/to/file
```

### Tree structure

`--mlir-print-ir-tree-dir` nests the dumped IR in a tree structure that matches
the nesting structure of the pass pipeline. In particular, by default all passes
will be nested under a `builtin_module_no-symbol-name/` folder, and any passes
in the pipeline that are anchored on nested operations like `func.func` will be
placed in subdirectories based on the name, such as
`func_func_matvec__decrypt__result0` for a pass anchored on `func.func` when the
IR contains a function called `matvec__decrypt__result0`.

For example, if the flag `--mlir-print-ir-tree-dir=/tmp/mlir` is used, then the
dumped files might look like:

```
/tmp/mlir/
└── builtin_module_no-symbol-name
    ├── 0_annotate-module.mlir
    ├── 10_compare-to-sign-rewrite.mlir
    ├── 11_canonicalize.mlir
    ├── 12_cse.mlir
    ├── 13_apply-folders.mlir
    ├── 14_canonicalize.mlir
    ├── 15_insert-rotate.mlir
    ├── 16_cse.mlir
    ├── 17_canonicalize.mlir
    ├── 18_cse.mlir
    ├── 19_collapse-insertion-chains.mlir
    ├── 1_debug-validate-names.mlir
    ├── 20_sccp.mlir
    ├── 21_canonicalize.mlir
    ...
    ├── func_func__assign_layout_9911323688340753831
    │   ├── 47_0_affine-expand-index-ops.mlir
    │   ├── 47_1_affine-simplify-structures.mlir
    │   ├── 47_2_affine-loop-normalize.mlir
    │   └── 47_3_forward-insert-slice-to-extract-slice.mlir
    ├── func_func_matvec
    │   ├── 47_0_affine-expand-index-ops.mlir
    │   ├── 47_1_affine-simplify-structures.mlir
    │   ├── 47_2_affine-loop-normalize.mlir
    │   └── 47_3_forward-insert-slice-to-extract-slice.mlir
    ├── func_func_matvec__decrypt__result0
    │   ├── 47_0_affine-expand-index-ops.mlir
    │   ├── 47_1_affine-simplify-structures.mlir
    │   ├── 47_2_affine-loop-normalize.mlir
    │   └── 47_3_forward-insert-slice-to-extract-slice.mlir
    └── func_func_matvec__encrypt__arg0
        ├── 47_0_affine-expand-index-ops.mlir
        ├── 47_1_affine-simplify-structures.mlir
        ├── 47_2_affine-loop-normalize.mlir
        └── 47_3_forward-insert-slice-to-extract-slice.mlir
```

### Dumped file contents

At the top of each file, a comment is printed indicating how the file was
generated, containing e.g., `IR Dump Before CSEPass (cse)`. This will say
whether the IR was dumped before or after a pass, as well as whether (in the
case of `--mlir-print-ir-after-failure`) the pass failed.

Below the comment, the MLIR is printed in textual format.

## Gotchas

- **Stale files**: `--mlir-print-ir-tree-dir` does not delete existing files, so
  if the command is re-run, be sure to remove the existing temporary files
  first.
- **Dump before vs after**: `--mlir-print-ir-before-all` prints the IR before a
  pass runs, which is required to extract the IR if a pass has a hard error like
  a segfault. To dump the IR *after* each pass, use `--mlir-print-ir-after-all`.
- **Dumping after failure**: For some errors without stack traces (like a
  dialect conversion failure where the compiler fails gracefully), you can use
  `--mlir-print-ir-after-failure` to dump the IR at failure time, which can
  provide additional debugging information. One notable example is when dialect
  conversion fails, the IR dumped by `--mlir-print-ir-after-failure` includes
  the partially-converted IR including any `unrealized_conversion_cast` ops that
  failed to be removed.
- **Dumping after change**: The number of files can be reduced by using
  `--mlir-print-ir-after-change`, which only prints the IR if the compiler pass
  changed it. There is no "before change" analogue.
- **Dumping after change**: The number of files can be reduced by using
  `--mlir-print-ir-before=<pass_name>`, which only prints the IR before the
  target pass. Since a pass can be run multiple times in a pipeline, this may
  still produce many files if, for example, the pass is a common one like `cse`.

```markdown
Copy this checklist and track progress:

- [ ] Step 1: Identify the command to convert, perhaps with `lit-to-bazel`
- [ ] Step 2: Add the relevant flags, defaulting to `--mlir-print-ir-before-all` and `--mlir-print-ir-tree-dir` if there is no specific reason to use the more specific flags.
- [ ] Step 3: Inspect the generated command and run it.
- [ ] Step 4: Inspect the dumped files for information.
```

<!-- mdformat global-off -->
