# heir_py Developer's Notes

This Python frontend is currently _experimental_. Current **missing** features
include:

- Being able to `pip install` the local development tree
- Being able to `pip install` from PyPI.

## Overview

The main complexity here is that the Python frontend generates MLIR, runs some
HEIR pipeline, and then may get as output language-specific source code (C++
code, Go code, Python code etc.), which then must be integrated back into the
frontend. For compiled languages, this requires a system-wide dependency on the
particular compiler toolchain and any installed library dependencies as shared
object files.

Bazel avoids these issues, with the exception of having a C++ compiler and the
C++ standard libraries available on you system.

## Running directly from Python

You should be able to run the frontend directly from python (without using
bazel) by (from the HEIR project root):

- `pip install -r heir_py/requirements.txt`
- `python -m heir_py.example` or `python -i -m heir_py.pipeline`

This will attempt to discover everything needed on your system from the
operating system.

You will need the following system-level dependencies to do this:

- `clang` (not sure what versions are supported, has been tested on `clang-17`)
- For C++ backends like OpenFHE, `python3.11-dev` (or similar for your python
  version) in order to build generated `pybind11` bindings.
- C/c++ standard libraries that come with your system's compiler toolchain, but
  may be in a nonstandard location. Ensure they can be discovered by a call to
  `clang` without any special flags. (`clang -v` will show you which paths it
  considers).

If this doesn't work, you may need to set up the following environment
variables:

- `HEIR_OPT_PATH` and `HEIR_TRANSLATE_PATH`: to the location of the `heir-opt`
  and `heir-translate` binaries on your system.
  - Uses `shutil.which` to find the binary on your path if not set.
  - Defaults to `bazel-bin/tools/heir-{opt,translate}`.
  - Cf. `heir_py/heir_config.py` for more details.
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

## Running from bazel

`bazel test` should work out of the box. If it does not, file a bug.
`heir_py/testing.bzl` contains the environment variable setup required to tell
the frontend where to find OpenFHE and related backend shared libraries.
