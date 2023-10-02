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
