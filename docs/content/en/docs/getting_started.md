<!-- mdformat off(yaml frontmatter) -->
---
title: Getting Started
weight: 1
---
<!-- mdformat on -->

## Prerequisites

-   [Git](https://git-scm.com/)
-   Bazel via [bazelisk](https://github.com/bazelbuild/bazelisk), or version
    `>=5.5`
-   A C compiler (like [gcc](https://gcc.gnu.org/) or
    [clang](https://clang.llvm.org/))

## Clone and build the project

```bash
git clone git@github.com:google/heir.git && cd heir
bazel build @heir//tools:heir-opt
```

Some passes in this repository require Yosys as a dependency
(`--yosys-optimizer`). If you would like to skip Yosys and ABC compilation to
speed up builds, use the following build setting:

```bash
bazel build --//:enable_yosys=0 @heir//tools:heir-opt
```

## Optional: Run the tests

```bash
bazel test @heir//...
```

Like above, run the following to skip tests that depend on Yosys:

```bash
bazel test --//:enable_yosys=0 --test_tag_filters=-yosys @heir//...
```

## Run the hello-world example

The `hello-world` example is a simple program that

## Optional: Run a custom `heir-opt` pipeline

HEIR comes with two central binaries, `heir-opt` for running optimization passes
and dialect conversions, and `heir-translate` for backend code generation. To
see the list of available passes in each one, run the binary with `--help`:

```bash
bazel run //tools:heir-opt -- --help
bazel run //tools:heir-translate -- --help
```

Once you've chosen a pass or `--pass-pipeline` to run, execute it on the desired
file. For example, you can run a test file through `heir-opt` to see its output.
Note that when the binary is run via `bazel`, you must pass absolute paths to
input files. You can also access the underlying binary at
`bazel-bin/tools/heir-opt`, provided it has already been built.

```bash
bazel run //tools:heir-opt -- \
  --comb-to-cggi -cse \
  $PWD/tests/comb_to_cggi/add_one.mlir
```

To convert an existing lit test to a `bazel run` command for manual tweaking
and introspection (e.g., adding `--debug` or `--mlir-print-ir-after-all` to see
how he IR changes with each pass), use `python scripts/lit_to_bazel.py`.

```bash
# after pip installing requirements-dev.txt
python scripts/lit_to_bazel.py tests/simd/box_blur_64x64.mlir
```

Which outputs

```bash
bazel run --noallow_analysis_cache_discard //tools:heir-opt -- \
--secretize=entry-function=box_blur --wrap-generic --canonicalize --cse --full-loop-unroll \
--insert-rotate --cse --canonicalize --collapse-insertion-chains \
--canonicalize --cse /path/to/heir/tests/simd/box_blur_64x64.mlir
```

## Developing in HEIR

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
