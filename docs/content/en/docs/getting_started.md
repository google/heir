<!-- mdformat off(yaml frontmatter) -->
---
title: Getting Started
weight: 1
---
<!-- mdformat on -->

## Prerequisites

-   [Git](https://git-scm.com/)
-   [Bazel](https://github.com/bazelbuild/bazelisk)
-   A C compiler (like [gcc](https://gcc.gnu.org/) or
    [clang](https://clang.llvm.org/))
-   [Python](https://www.python.org/) (optional, for some tests)

## Clone and build the project

```bash
git clone git@github.com:google/heir.git && cd heir
bazel build @heir//...
```

## Optional: Run the tests

```bash
bazel test @heir//...
```

Note some tests require Python to run.
