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
