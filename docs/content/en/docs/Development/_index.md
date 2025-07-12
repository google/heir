---
title: Development
weight: 30
---

## Pre-Commit

We use [pre-commit](https://pre-commit.com/) to manage a series of git
pre-commit hooks for the project; for example, each time you commit code, the
hooks will make sure that your C++ is formatted properly. If your code isn't,
the hook will format it, so when you try to commit the second time you'll get
past the hook. Note that spelling mistakes identified by the codespell hook will
not be auto-corrected and require manual resolution, rather than simply
re-running pre-commit.

All hooks are defined in `.pre-commit-config.yaml`. To install these hooks, first run

```bash
pip install -r requirements.txt
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

## Tips for building dependencies / useful external libraries

Sometimes it is useful to point HEIR to external dependencies built according to
the project's usual build system, instead of HEIR's bazel overlay. For example,
to test upstream contributions to the dependency in the context of how it will
be used in HEIR.

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
Intel [HEXL library](https://github.com/intel/hexl) which adds AVX512 support.
First, clone the repository
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
