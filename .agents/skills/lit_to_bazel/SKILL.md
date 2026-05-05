---
name: lit-to-bazel
description: >-
  Converts MLIR lit test files to `bazel run` commands. Use when you need to
  convert a failing MLIR lit test target (containing RUN lines) to a shell
  command whose flags can be modified for further analysis and debugging.
---

# Lit To Bazel

## Overview

This skill guides the agent in using the `lit_to_bazel` tool to convert MLIR
test files to executable commands.

## Usage

### Converting a Test File

To convert a lit test file to a `bazel run` command, use the following command
recipes:

```bash
bazel run //third_party/heir/scripts:lit_to_bazel -- {test_file_path}
```

Replace `{test_file_path}` with the absolute path to the test file. The absolute
path is necessary because the `bazel run` command runs relative to its runfiles
directory, and so a relative path will not be relative to the VCS root of the
project.

## Gotchas

- **Absolute Paths**: The tool replaces `%s` in RUN lines with the absolute path
  of the test file to ensure it works in the sandbox.
- **Piped RUN commands**: The tool supports piped RUN lines, such as piping
  heir-opt into heir-translate.
- **Complex RUN commands**: Some very complex RUN lines are not supported, such
  as lit test files with multiple RUN lines that interact via writing to and
  reading from temporary files. When encountering situations like this, you
  should stop and warn the user.
- **Non-lit tests**: Some test targets are not lit tests, and these do not need
  the `lit_to_bazel` tool.

```markdown
Copy this checklist and track progress:

- [ ] Step 1: Identify the test file to convert.
- [ ] Step 2: Run the `lit_to_bazel` tool on the file.
- [ ] Step 3: Inspect the generated command or run it.
- [ ] Step 4: Verify execution results.
```

<!-- mdformat global-off -->
