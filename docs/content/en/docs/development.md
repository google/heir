<!-- mdformat off(yaml frontmatter) -->
---
title: Development
weight: 30
---
<!-- mdformat on -->

## IDE Configuration (VS Code)
While a wide variety of IDEs and editors
can be used for HEIR development, we
currently only provide support
for [VSCode](https://code.visualstudio.com/).

### Setup

For the best experience, we recommend following these steps:

* Install the [MLIR](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-mlir),
   [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd)
    and [Bazel](https://marketplace.visualstudio.com/items?itemName=BazelBuild.vscode-bazel) extensions

* Install and rename Buildifier:

  You can download the latest Buildifier release, e.g.,
  for linux-amd64 (see the
  [Bazelisk Release Page](https://github.com/bazelbuild/buildtools/releases/latest/)
  for a list of available binaries):
    ```bash
    wget -c https://github.com/bazelbuild/buildtools/releases/latest/download/buildifier-linux-amd64
    mv buildifier-linux-amd64 buildifier
    chmod +x buildifier
    ```

    Just as with bazel, you will want to move this somewhere on your PATH, e.g.:
    ```bash
    mkdir ~/bin
    echo 'export PATH=$PATH:~/bin' >> ~/.bashrc
    mv buildifier ~/bin/buildifier
    ```

    VS Code should automatically detect buildifier.
    If this is not successful, you can manually set
    the "Buildifier Executable" setting
    for the Bazel extension (`bazel.buildifierExecutable`).

* Disable the [C/C++ (aka 'cpptools')](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) extension (either completely, or in the current workspace).

* Add the following snippet to your VS Code user settings
found in .vscode/settings.json to enable autocomplete
based on the compile_commands.json file.
  ```json
    "clangd.arguments": [
      "--compile-commands-dir=${workspaceFolder}/",
      "--completion-style=detailed",
      "--query-driver=**"
    ],
  ```

* To generate the `compile_commands.json` file, run
  ```shell
  bazel run @hedron_compile_commands//:refresh_all
  ```

  This will need to be regenerated every time you want tooling to see new `BUILD`
  file changes.

  If you encounter errors like `*.h.inc` not found, or syntax errors inside these
  files, you may need to build those targets and then re-run the `refresh_all`
command above.


* NOTE: In order to share bazel's build cache between invocations from VScode's GUI and the terminal, you need to create a `.bazelrc` in your home directory that sets the compilation toolchain to clang. Alternatively, you can set this in your vscode settings to only apply to the current workspace, but this means switching between compilation from terminal and from VSCode will trigger rebuilds.
    ```bash
    echo -e "# enforces clang use\ncommon --repo_env=CC=clang" > ~/.bazelrc
    ```
*  It might be necesssary to add the path to your buildifier to VSCode, though it should be auto-detected.
    - Open the heir folder in VSCode
    - Go to 'Settings' and set it on the 'Workspace'
    - Search for "Bazel Buildifier Executable"
    - Once you find it, write ```[home-directory]/bin/buildifier ``` for your specific [home-directory].

### Building, Testing, Running and Debugging with VSCode

#### Building
1. Open the "Explorer" (File Overview) in the left panel.
1. Find "Bazel Build Targets" towards the bottom of the "Explorer" panel and click the dropdown button.
1. Unfold the heir folder
1. Right-click on "//tools" and click the "Build Package Recursively" option

#### Testing
1. Open the "Explorer" (File Overview) in the left panel.
1. Find "Bazel Build Targets" towards the bottom of the "Explorer" panel and click the dropdown button.
1. Unfold the heir folder
1. Right-click on "//test" and click the "Test Package Recursively" option

#### Running and Debugging
1. Create a `launch.json` file in the `.vscode` folder, changing the `"name"` and `"args"` as required:
    ```json
    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Debug Secret->BGV",
                "preLaunchTask": "build",
                "type": "lldb",
                "request": "launch",
                "program": "${workspaceFolder}/bazel-bin/tools/heir-opt",
                "args": [
                    "--secret-to-bgv",
                    "--debug",
                    "${workspaceFolder}/tests/secret_to_bgv/ops.mlir"
                ],
                "relativePathBase": "${workspaceFolder}",
                "sourceMap": {
                    "proc/self/cwd": "${workspaceFolder}",
                    "/proc/self/cwd": "${workspaceFolder}"
                }
            },
        ]
   }
   ```
    You can add as many different configurations as necessary.

1. Add Breakpoints to your program as desired.
1. Open the Run/Debug panel on the left, select the desired configuration and run/debug it.
  * Note that you might have to hit "Enter" to proceed past the Bazel build.
    It might take several seconds between hitting "Enter" and the debug terminal opening.

## Tips for working with Bazel

### Avoiding rebuilds

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

### Pointing HEIR to a local clone of `llvm-project`

Occasionally changes in HEIR will need to be made in tandem with upstream
changes in MLIR. In particular, we occasionally find upstream bugs that only
occur with HEIR passes, and we are the primary owners/users of the upstream
`polynomial` dialect.

To tell `bazel` to use a local clone of `llvm-project` instead of a pinned
commit hash, replace `bazel/import_llvm.bzl` with the following file:

```bash
cat > bazel/import_llvm.bzl << EOF
"""Provides the repository macro to import LLVM."""

def import_llvm(name):
    """Imports LLVM."""
    native.new_local_repository(
        name = name,
        # this BUILD file is intentionally empty, because the LLVM project
        # internally contains a set of bazel BUILD files overlaying the project.
        build_file_content = "# empty",
        path = "/path/to/llvm-project",
    )
EOF
```

The next `bazel build` will require a full rebuild if the checked-out LLVM
commit differs from the pinned commit hash in `bazel/import_llvm.bzl`.

Note that you cannot reuse the LLVM CMake build artifacts in the bazel build.
Based on what you're trying to do, this may require some extra steps.

- If you just want to run existing MLIR and HEIR tests against local `llvm-project`
  changes, you can run the tests from HEIR using `bazel test @llvm-project//mlir/...:all`.
  New `lit` tests can be added in `llvm-project`'s existing directories and
  tested this way without a rebuild.
- If you add new CMake targets in `llvm-project`, then to incorporate them into
  HEIR you need to add new bazel targets in
  `llvm-project/utils/bazel/llvm-project-overlay/mlir/BUILD.bazel`. This is
  required if, for example, a new dialect or pass is added in MLIR upstream.

Send any upstream changes to HEIR-relevant MLIR files to @j2kun (Jeremy Kun)
who has LLVM commit access and can also suggest additional MLIR reviewers.


## Creating a New Pass

The `scripts/templates` folder contains Python scripts to create boilerplate
for new conversion or (dialect-specific) transform passes. These should be used
when the tablegen files containing existing pass definitions in the expected
filepaths are not already present. Otherwise, you should modify the existing
tablegen files directly.

### Conversion Pass

To create a new conversion pass, run a command similar to the following:

```
python scripts/templates/templates.py new_conversion_pass \
--source_dialect_name=CGGI \
--source_dialect_namespace=cggi \
--source_dialect_mnemonic=cggi \
--target_dialect_name=TfheRust \
--target_dialect_namespace=tfhe_rust \
--target_dialect_mnemonic=tfhe_rust
```

In order to build the resulting code, you must fix the labeled `FIXME`s in the
type converter and the op conversion patterns.

### Transform Passes

To create a transform or rewrite pass that operates on a dialect, run a command
similar to the following:

```
python scripts/templates/templates.py new_dialect_transform \
--pass_name=ForgetSecrets \
--pass_flag=forget-secrets \
--dialect_name=Secret \
--dialect_namespace=secret \
--force=false
```

If the transform does not operate from and to a specific dialect, use

```
python scripts/templates/templates.py new_transform \
--pass_name=ForgetSecrets \
--pass_flag=forget-secrets \
--force=false
```


## Pre-Commit
We use [pre-commit](https://pre-commit.com/) to manage a series of git
pre-commit hooks for the project; for example, each time you commit code, the
hooks will make sure that your C++ is formatted properly. If your code isn't,
the hook will format it, so when you try to commit the second time you'll get
past the hook.

All hooks are defined in `.pre-commit-config.yaml`. To install these hooks, run

```bash
pip install -r requirements-dev.txt
```

Then install the hooks to run automatically on `git commit`:

```bash
pre-commit install
```

To run them manually, run

```bash
pre-commit run --all-files
```
