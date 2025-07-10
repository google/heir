---
title: Development
weight: 30
---

## IDE Configuration (VS Code)

While a wide variety of IDEs and editors can be used for HEIR development, we
currently only provide support for [VSCode](https://code.visualstudio.com/).

### Setup

For the best experience, we recommend following these steps:

- Install the
  [MLIR](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-mlir),
  [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd)
  and
  [Bazel](https://marketplace.visualstudio.com/items?itemName=BazelBuild.vscode-bazel)
  extensions

- Install and rename Buildifier:

  You can download the latest Buildifier release, e.g., for linux-amd64 (see the
  [Bazelisk Release Page](https://github.com/bazelbuild/buildtools/releases/latest/)
  for a list of available binaries):

  ```bash
  wget -c https://github.com/bazelbuild/buildtools/releases/latest/download/buildifier-linux-amd64
  mv buildifier-linux-amd64 buildifier
  chmod +x buildifier
  ```

  Just as with bazel, you will want to move this somewhere on your PATH, e.g.:

  ```bash
  mkdir -p ~/bin
  echo 'export PATH=$PATH:~/bin' >> ~/.bashrc
  mv buildifier ~/bin/buildifier
  ```

  VS Code should automatically detect buildifier. If this is not successful, you
  can manually set the "Buildifier Executable" setting for the Bazel extension
  (`bazel.buildifierExecutable`).

- Disable the
  [C/C++ (aka 'cpptools')](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)
  extension (either completely, or in the current workspace).

- Add the following snippet to your VS Code user settings found in
  .vscode/settings.json to enable autocomplete based on the
  compile_commands.json file.

  ```json
    "clangd.arguments": [
      "--compile-commands-dir=${workspaceFolder}/",
      "--completion-style=detailed",
      "--query-driver=**"
    ],
  ```

- To generate the `compile_commands.json` file, run

  ```shell
  bazel run @hedron_compile_commands//:refresh_all
  ```

  This will need to be regenerated every time you want tooling to see new
  `BUILD` file changes.

  If you encounter errors like `*.h.inc` not found, or syntax errors inside
  these files, you may need to build those targets and then re-run the
  `refresh_all` command above.

- It might be necesssary to add the path to your buildifier to VSCode, though it
  should be auto-detected.

  - Open the heir folder in VSCode
  - Go to 'Settings' and set it on the 'Workspace'
  - Search for "Bazel Buildifier Executable"
  - Once you find it, write `[home-directory]/bin/buildifier ` for your specific
    \[home-directory\].

### Building, Testing, Running and Debugging with VSCode

#### Building

1. Open the "Explorer" (File Overview) in the left panel.
1. Find "Bazel Build Targets" towards the bottom of the "Explorer" panel and
   click the dropdown button.
1. Unfold the heir folder
1. Right-click on "//tools" and click the "Build Package Recursively" option

#### Testing

1. Open the "Explorer" (File Overview) in the left panel.
1. Find "Bazel Build Targets" towards the bottom of the "Explorer" panel and
   click the dropdown button.
1. Unfold the heir folder
1. Right-click on "//test" and click the "Test Package Recursively" option

#### Running and Debugging

1. Create a `launch.json` file in the `.vscode` folder, changing the `"name"`
   and `"args"` as required:

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

1. Open the Run/Debug panel on the left, select the desired configuration and
   run/debug it.

- Note that you might have to hit "Enter" to proceed past the Bazel build. It
  might take several seconds between hitting "Enter" and the debug terminal
  opening.

## Tips for working with Bazel

### Avoiding rebuilds

Bazel is notoriously fickle when it comes to deciding whether a full rebuild is
necessary, which is bad for HEIR because rebuilding LLVM from scratch takes 15
minutes or more. We try to avoid this as much as possible by setting default
options in the project root's `.bazelrc`.

The main things that cause a rebuild are:

- A change to the `.bazelrc` that implicitly causes a flag change. Note HEIR has
  its own project-specific `.bazelrc` in the root directory.
- A change to the command-line flags passed to bazel, e.g., `-c opt` vs `-c dbg`
  for optimization level and debug symbols. The default is `-c dbg`, and you may
  want to override this to optimize performance of generated code. For example,
  the OpenFHE backend generates much faster code when compiled with `-c opt`.
- A change to relevant command-line variables, such as `PATH`, which is avoided
  by the `incompatible_strict_action_env` flag. Note activating a python
  virtualenv triggers a `PATH` change. The default is
  `incompatible_strict_action_env=true`, and you would override this in the
  event that you want your shell's environment variables to change and be
  inherited by bazel.

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

- If you just want to run existing MLIR and HEIR tests against local
  `llvm-project` changes, you can run the tests from HEIR using
  `bazel test @llvm-project//mlir/...:all`. New `lit` tests can be added in
  `llvm-project`'s existing directories and tested this way without a rebuild.
- If you add new CMake targets in `llvm-project`, then to incorporate them into
  HEIR you need to add new bazel targets in
  `llvm-project/utils/bazel/llvm-project-overlay/mlir/BUILD.bazel`. This is
  required if, for example, a new dialect or pass is added in MLIR upstream.

Send any upstream changes to HEIR-relevant MLIR files to @j2kun (Jeremy Kun) who
has LLVM commit access and can also suggest additional MLIR reviewers.

### Finding the right dependency targets

Whenever a new dependency is added in C++ or Tablegen, a new bazel BUILD
dependency is required, which requires finding the path to the relevant target
that provides the file you want. In HEIR the BUILD target should be defined in
the same directory as the file in question, but upstream MLIR's bazel layout is
different.

LLVM's bazel overlay for MLIR is contained in a
[single file](https://github.com/llvm/llvm-project/blob/main/utils/bazel/llvm-project-overlay/mlir/BUILD.bazel),
and so you can manually look there to find the right target. With bazel, if you
know the filepath of interested you can also run:

```bash
bazel query --keep_going 'same_pkg_direct_rdeps(@llvm-project//mlir:<path>)'
```

where `<path>` is the path relative to `mlir/` in the `llvm-project` project
root. For example, to find the target that provides
`mlir/include/mlir/Pass/PassBase.td`, run

```bash
bazel query --keep_going 'same_pkg_direct_rdeps(@llvm-project//mlir:include/mlir/Pass/PassBase.td)'
```

And the output will be something like

```bash
@llvm-project//mlir:PassBaseTdFiles
```

You can find more examples and alternative queries at the
[Bazel query docs](https://bazel.build/query/language#rdeps).

## Tips for building dependencies / useful external libraries

### MLIR

Instructions for building MLIR can be found on the
[Getting started](https://mlir.llvm.org/getting_started/) page of the MLIR
website. The instructions there seem to work as written (tested on Ubuntu
22.04). However, the command shown in `Unix-like compile/testing:` may require a
large amount of RAM. If building on a system with 16GB of RAM or less, and if
you don't plan to target GPUs, you may want to replace the line

```
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
```

with

```
   -DLLVM_TARGETS_TO_BUILD="Native" \
```

### OpenFHE

A simple way to build OpenFHE is to follow the instructions in the
[openfhe-configurator](https://github.com/openfheorg/openfhe-configurator)
repository. This allows to build the library with or without support for the
Intel [HEXL library](https://github.com/intel/hexl). First, clone the repository
and configure it using:

```
git clone https://github.com/openfheorg/openfhe-configurator.git
cd openfhe-configurator
scripts/configure.sh
```

You will be asked whether to stage a vanilla OpenFHE build or add support for
HEXL. You can then build the library using

```
./scripts/build-openfhe-development.sh
```

The build may fail on systems with less than 32GB or RAM due to parallel
compilation. You can disable it by editing
`./scripts/build-openfhe-development.sh` and replacing

```
make -j || abort "Build of openfhe-development failed."
```

with

```
make || abort "Build of openfhe-development failed."
```

Compilation will be significantly slower but should then take less than 8GB of
memory.

## Creating a New Pass

The `scripts/templates` folder contains Python scripts to create boilerplate for
new conversion or (dialect-specific) transform passes. These should be used when
the tablegen files containing existing pass definitions in the expected
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
past the hook. Note that spelling mistakes identified by the codespell hook will
not be auto-corrected and require manual resolution, rather than simply
re-running pre-commit.

All hooks are defined in `.pre-commit-config.yaml`. To install these hooks, run

```bash
pip install -r requirements-dev.txt
```

You will also need to install ruby and go (e.g., `apt-get install ruby golang`)
which are used by some of the pre-commits. Note that the pre-commit environment
expects Python 3.11
([Installing python3.11 on ubuntu](https://askubuntu.com/a/1512163)).

Then install the hooks to run automatically on `git commit`:

```bash
pre-commit install
```

To run them manually, run

```bash
pre-commit run --all-files
```
