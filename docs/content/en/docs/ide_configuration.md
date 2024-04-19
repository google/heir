<!-- mdformat off(yaml frontmatter) -->
---
title: IDE Configuration
weight: 3
---
<!-- mdformat on -->

## VS Code

For an out-of-tree MLIR project built with Bazel, install the following VS Code
extensions:

- **llvm-vs-code-extensions.vscode-mlir**: Adds language support for MLIR, PDLL,
  and TableGen.
- **llvm-vs-code-extensions.vscode-clangd**: Adds clangd code completion using a
  generated
  [compile_commands.json](https://clang.llvm.org/docs/JSONCompilationDatabase.html)
  file.
- **bazelbuild.vscode-bazel**: Support for Bazel.

You will also need to disable **ms-vscode.cpptools** to avoid a conflict with
clangd.

Add the following snippet to your VS Code user settings found in
`.vscode/settings.json` to enable autocomplete based on the
`compile_commands.json` file.

```json
   "clangd.arguments": [
        "--compile-commands-dir=${workspaceFolder}/",
        "--completion-style=detailed",
        "--query-driver=**"
      ],
```

To generate the `compile_commands.json` file, run

```shell
bazel run @hedron_compile_commands//:refresh_all
```

This will need to be regenerated every time you want tooling to see new `BUILD`
file changes.

If you encounter errors like `*.h.inc` not found, or syntax errors inside these
files, you may need to build those targets and then re-run the `refresh_all`
command above.

## Tips for working with Bazel

Bazel is notoriously fickle when it comes to deciding whether a full rebuild is
necessary, which is bad for HEIR because rebuilding LLVM from scratch takes 15
minutes or more.

The main things that cause a rebuild are:

-   A change to the command-line flags passed to bazel, e.g., `-c opt` vs `-c
    dbg` for optimization level and debug symbols.
-   A change to the `.bazelrc` that implicitly causes a flag change. Note HEIR
    has its own project-specific `.bazelrc` in the root directory.
-   A change to relevant command-line variables, such as `PATH`, which is
    avoided by the `incompatible_strict_action_env` flag. Note activating a
    python virtualenv triggers a `PATH` change.

Bazel compilation flags are set by default in the project root's `.bazelrc` in
such a way as to avoid rebuilds during development as much as possible. This
includes setting `-c dbg` and `--incompatible_strict_action_env`.
