---
title: Bazel tips
weight: 30
---

## BUILD file formatting

The `buildifier` tool can be used to format BUILD files. You can download the
latest Buildifier release from the
[Bazel Release Page](https://github.com/bazelbuild/buildtools/releases/latest/).
See [IDE configuration](/docs/development/ide/) for tips on integrating this
with your IDE.

## Avoiding rebuilds

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

## Pointing HEIR to a local clone of `llvm-project`

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

## Finding the right dependency targets

Whenever a new dependency is added in C++ or Tablegen, a new bazel BUILD
dependency is required, which requires finding the path to the relevant target
that provides the file you want. In HEIR the BUILD target should be defined in
the same directory as the file you want to depend on (e.g., the targets that
provide `foo.h` are in `BUILD` in the same directory), but upstream MLIR's bazel
layout is different.

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

<!-- mdformat global-off -->
