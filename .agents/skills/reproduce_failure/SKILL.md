---
name: reproduce-failure
description: >-
  Describes a decision tree of steps and skills to utilize when, starting from a
  test failure or the failure of `bazel run` command using heir-opt, you would
  like to produce a reproducing input IR that fails on a particular compiler
  pass.
---

# Reproduce a test failure

## Overview

This skill guides the agent in identifying an IR and compiler pass that
reproduces a failure from a lit or e2e test target, or from a `bazel run`
command that runs `heir-opt`.

## Usage

To reproduce a failure, follow these steps.

### Identify the starting point of the process

The user may input one of multiple "starting points" for a failure.

1. A direct `bazel run` command.
1. A `lit` test target or `bazel test` command pointing to such a target, such
   as
   `//tests/Transforms/convert_to_ciphertext_semantics:assign_layout.mlir.test`,
1. A file path to a lit test, such as
   `third_party/heir/tests/Transforms/convert_to_ciphertext_semantics/assign_layout.mlir`
1. An end-to-end test target, such as
   `//tests/Examples/lattigo/ckks/dot_product_8f:dotproduct8f_test`,
1. A filepath to an end-to-end test file, such as
   `third_party/heir/tests/Examples/lattigo/ckks/dot_product_8f/dot_product_8f_test.go`

Reiterate to the user that you understand which of these options are the
starting point, and if the user provides a starting point that doesn't fit into
one of these options, warn them and then ask for confirmation when improvising
how to convert their failure to a `bazel run` command in the next step.

### Identify the right `bazel run` command

For option 1 (A direct `bazel run` command), nothing is needed, continue to the
next step.

For option 2, use the `lit_to_bazel` skill.

For option 3, use the `e2e_to_bazel` skill.

At the end of this step, you should have a `bazel run` command that reproduces
the user's error, but that command may run many compiler passes in a pipeline.
The remaining steps will reduce the `bazel run` command to a more minimal
reproducer.

Run it to ensure it reproduces the user's error.

### Identify the specific IR, pass, and pass options

Use the `dump_intermediate_ir` skill to augment the `bazel run` command with
appropriate flags, so that it writes relevant files that contain the IR and pass
options.

Then inspect the right file (usually the last file dumped before an error
occurs) to identify the IR and pass options. The dumped IR should include near
the top of the file, a comment like

```mlir
// -----// IR Dump Before CanonicalizerPass: canonicalize{cse-between-iterations=false max-iterations=5 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true} //----- //
```

The part after `:` is the relevant pass name and options that will be needed in
the next step. The rest of the file content is the IR that will be needed in the
next step.

### Construct the `bazel run` command for the individual pass

Given the information from previous steps, construct a `bazel run` command of
the form:

```bash
bazel run //tools:heir-opt -- \
--pass-pipeline="<PASS_AND_OPTIONS>" \
<IR_FILE>
```

where `<PASS_AND_OPTIONS>` corresponds to the "pass name and options" from the
previous step, and `<IR_FILE>` corresponds to the dumped IR for that pass.

At this step, run the new `bazel run` command and make sure that it reproduces
the error.

### Create a reproducing `lit` test file

Copy the IR to a new `lit` regression test file with the `bazel run` command
modified to a `// RUN:` line.

The new test should live in `third_party/heir/tests/Regression/` and have at the
top of the file a comment describing the origin of the reproducer along with the
`// RUN:` line.

For example, if the reduced `bazel run` command from the previous step was

```bash
bazel run //tools:heir-opt -- \
--pass-pipeline="canonicalze{cse-between-iterations=true}" \
/tmp/mlir/foo.mlir
```

and the original command was
`bazel run //tools:heir-opt -- --big-pipeline /path/to/file.mlir`

Then the corresponding lit file should start with

```mlir
// This file is a minimal reproducer of a failure originally produced with
//
//    heir-opt --big-pipeline /path/to/file.mlir
//
// RUN: heir-opt --pass-pipeline="canonicalze{cse-between-iterations=true}" %s
```

Below that header, place the reproducing IR from the previous step. Then confirm
the test exercises the test failure once more by running
`bazel test //tests/Regression:<name_of_file.mlir>.test`.

The `<name_of_file.mlir>` should be chosen appropriately according to the
following heuristic:

- If you know of a particular GitHub issue number, use the number, like
  `issue_1480.mlir`.
- If you have a starting test file or target, name it similar to that file,
  e.g., `dot_product_8f_regression.mlir`
- If it makes sense, incorporate the name of the pass, e.g.,
  `dot_product_secret_to_ckks.mlir`
- If no natural name makes sense, use `RENAME_ME_regression_test.mlir`

## Gotchas

- Ensure that at each step you can still reproduce the failure. If at any step,
  a newly reduced command fails to reproduce the failure, stop and report this
  to the user, asking for assistance.

```markdown
Copy this checklist and track progress:

- [ ] Step 1: Identify the starting point of the process
- [ ] Step 2: Identify the right `bazel run` command
- [ ] Step 3: Identify the specific IR, pass, and pass options
- [ ] Step 4: Construct the `bazel run` command for the individual pass
- [ ] Step 5: Create a reproducing `lit` test file

Each step corresponds to a section mentioned above.
```

<!-- mdformat global-off -->
