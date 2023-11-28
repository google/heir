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

Some passes in this repository require Yosys as a dependency (`--yosys-optimizer`). If you would like to skip Yosys and ABC compilation to speed up builds, use the following build setting:

```bash
bazel build --define=HEIR_NO_YOSYS=1 @heir//tools:heir-opt
```

## Optional: Run the tests

```bash
bazel test @heir//...
```

Like above, run the following to skip tests that depend on Yosys:


```bash
bazel test --define=HEIR_NO_YOSYS=1 --test_tag_filters=-yosys @heir//...
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

The `templates` folder contains Python scripts to create boilerplate for new conversion or (dialect-specific) transform passes.

### Conversion Pass

To create a new conversion pass, run a command similar to the following:

```
python templates/templates.py new_conversion_pass \
--source_dialect_name=CGGI \
--source_dialect_namespace=cggi \
--source_dialect_mnemonic=cggi \
--target_dialect_name=TfheRust \
--target_dialect_namespace=tfhe_rust \
--target_dialect_mnemonic=tfhe_rust
```

In order to build the resulting code, you must fix the labeled `FIXME`s in the type converter and the op conversion patterns.

### Transform Passes

To create a transform or rewrite pass that operates on a dialect, run a command similar to the following:

```
python templates/templates.py new_dialect_transform \
--pass_name=ForgetSecrets \
--pass_flag=forget-secrets \
--dialect_name=Secret \
--dialect_namespace=secret \
--force=false
```

If the transform does not operate from and to a specific dialect, use

```
python templates/templates.py new_transform \
--pass_name=ForgetSecrets \
--pass_flag=forget-secrets \
--force=false
```
