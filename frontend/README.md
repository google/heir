# HEIR Frontend Developer's Notes

## Overview

The main complexity here is that the Python frontend generates MLIR, runs some
HEIR pipeline, and then may get as output language-specific source code (C++
code, Go code, Python code etc.), which then must be integrated back into the
frontend. For compiled languages, this requires a system-wide dependency on the
particular compiler toolchain and any installed library dependencies as shared
object files.

Bazel avoids these issues, with the exception of having a C++ compiler and the
C++ standard libraries available on you system.

## Local development

System requirements:

- A C compiler, such as `clang++` or `g++`.
- C/c++ standard libraries that come with the compiler, but may be in a
  nonstandard location. Ensure they can be discovered by a call to `clang`
  without any special flags. (`clang -v` will show you which paths it
  considers).
- Python development headers (e.g., `python3.11-dev` or similar).

### Running with bazel

1. Use the macros in `testing.bzl` to create `py_test` rules that exercise the
   frontend, and run them with `bazel test`.
1. Create `py_binary` rules depending on `@heir//frontend` to create
   executables, and run then with `bazel run`.

### Local editable pip installation

1. Ensure `bazel build //tools:all` has been run to build `heir-opt` and
   `heir-translate`.
1. Create a virtualenv: `python3.11 -m venv venv`. It should also work with
   Python 3.12 and 3.13.
1. Install the frontend: `pip install -e .` (if this fails, add `-v`). In this
   case, the installed package will autodetect paths to relevant resources from
   the bazel build.

### Local pip installation

1. Install: `pip install .`. This will run the bazel build, copy relevant files,
   and install the package just like a wheel installed from PyPI.

## Environment Variables

The Python frontend uses the following environment variables as overrides for
auto-detected resources.

- `HEIR_OPT_PATH` and `HEIR_TRANSLATE_PATH`: to the location of the `heir-opt`
  and `heir-translate` binaries on your system.

  - Uses `shutil.which` to find the binary on your path if not set.
  - Defaults to `bazel-bin/tools/heir-{opt,translate}`.
  - Cf. `heir/backends/openfhe/config.py` for more details.

- OpenFHE installation locations (default to where `cmake` installs them in the
  OpenFHE development repo).

  - `OPENFHE_LIB_DIR`: a string containing the directory containing the OpenFHE
    .so files. Usually `/usr/local/lib`
  - `OPENFHE_INCLUDE_DIR`: a colon-separated string of directories containing
    OpenFHE headers. Note this usually requires four different paths due to how
    OpenFHE organizes its imports, one for each of the three main subdirectories
    of the project.
    - `/usr/local/include/openfhe`
    - `/usr/local/include/openfhe/binfhe`
    - `/usr/local/include/openfhe/core`
    - `/usr/local/include/openfhe/pke`
  - `OPENFHE_LINK_LIBS`: a colon-separated string of libraries to link against
    (without `lib` or `.so`). E.g., `"OPENFHEbinfhe:OPENFHEcore:OPENFHEpke"`.
  - `OPENFHE_INCLUDE_TYPE`: a string indicating the include path type to use
    (see options on `heir-translate --emit-openfhe`). Should be
    `install-relative` for a system-wide OpenFHE installation.

## Formatting

This uses [pyink](https://github.com/google/pyink) for autoformatting, which is
a fork of the more commonly used [black](https://github.com/psf/black) formatter
with some patches to support Google's internal style guide. The configuration in
pyproject.toml corresponds to the options making pyink consistent with Google's
internal style guide.

The `pyink` repo has instructions for setting up `pyink` with VSCode. The
pre-commit configuration for this repo will automatically run `pyink`, and to
run a one-time format of the entire project, use `pre-commit run --all-files`.

## Building wheels

The HEIR Python package is built using
[cibuildwheel](https://cibuildwheel.pypa.io/), which is also what CI uses to
produce PyPI releases. To test wheel builds locally:

### Prerequisites

- Install docker or podman (required for containerized linux builds). If using
  podman, set the environment variable `CIBW_CONTAINER_ENGINE=podman`.
- Install `cibuildwheel` (`pip install cibuildwheel`).
- **macOS only**: cibuildwheel requires the official
  [python.org](https://www.python.org/downloads/) framework installers (not
  Homebrew or uv-managed Pythons). Download and install the `.pkg` for each
  Python version you want to build (3.12, 3.13, etc.).

### Manually building a wheel

Build a single wheel (recommended for local testing):

```bash
cibuildwheel --only cp312-macosx_arm64      # macOS Apple Silicon
cibuildwheel --only cp312-macosx_x86_64     # macOS Intel
cibuildwheel --only cp312-manylinux_x86_64  # Linux x86_64
cibuildwheel --only cp312-manylinux_aarch64 # Linux arm64
```

The built wheel will be in `./wheelhouse/`. Note that building wheels for
non-native architectures will require emulation and therefore be significantly
slower.

### CI

The GitHub Actions workflow (`.github/workflows/wheels.yml`) builds wheels for
both Linux and macOS and uploads them to PyPI on release. It can also be
triggered manually via `workflow_dispatch`.

<!-- mdformat global-off -->
