---
name: e2e-to-bazel
description: >-
  Converts end-to-end (e2e) test targets or paths to `bazel run` commands for
  `heir-opt`. Use when you need to convert an e2e test target (where flags are
  defined in the BUILD file) to a shell command whose flags can be modified
  for further analysis and debugging.
---

# E2E To Bazel

## Overview

This skill guides the agent in using the `e2e_to_bazel` tool to convert e2e test
targets or paths to executable commands.

## Usage

### Converting a Test Target or Path

To convert an e2e test target, directory, or source file to a `bazel run`
command, use the following command recipes:

```bash
bazel run //scripts:e2e_to_bazel -- {target_or_path}
```

Replace `{target_or_path}` with the test target (e.g.,
`//tests/Examples/lattigo/ckks/mnist:mnist_test`), directory, or source file
path.

## Gotchas

- **Blaze Query Dependency**: The tool relies on `blaze query` to extract
  attributes. This requires a working `blaze` environment and may be slow for
  the first run.
- **Workspace State**: It requires that the workspace is in a state where
  `blaze query` can evaluate the targets.
- **Source file mapping**: If a source file is provided, it searches for
  `heir_opt` targets that depend on it.

```markdown
Copy this checklist and track progress:

- [ ] Step 1: Identify the e2e test target, directory, or file to convert.
- [ ] Step 2: Run the `e2e_to_bazel` tool on it.
- [ ] Step 3: Inspect the generated command or run it.
- [ ] Step 4: Verify execution results.
```

<!-- mdformat global-off -->
