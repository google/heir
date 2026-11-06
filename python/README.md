# HEIR Python package

This directory contains the `heir` Python package, published to PyPI as
[`heir_py`](https://pypi.org/project/heir-py/). It contains:

- HEIR's [MLIR Python bindings](https://mlir.llvm.org/docs/Bindings/Python/):
  the upstream MLIR Python API (`heir.ir`, `heir.passmanager`, ...), Python op
  builders for upstream dialects (`heir.dialects.arith`, `heir.dialects.func`,
  ...) and all HEIR dialects (`heir.dialects.secret`, `heir.dialects.bgv`,
  `heir.dialects.openfhe`, ...).
- A thin convenience layer (`heir.context`, `heir.parse`, `heir.run`,
  `heir.translate`, `heir.types`, ...) for running `heir-opt` passes and
  pipelines and `heir-translate` code emitters in-process.
- The `heir-opt` and `heir-translate` command line tools (in the wheel only).

The package requires Python 3.12+. Its native extensions are built against
CPython's stable ABI, so a single wheel per platform supports Python 3.12 and
all later versions.

## Example

```python
import heir
from heir import types
from heir.dialects import arith, func, secret

with heir.context():
  # Parse MLIR, or build it with the op builders (see below).
  module = heir.parse_file("dot_product_8.mlir")

  # Run passes and pipelines, as with heir-opt. Passes can be given as
  # heir.Pass objects, as textual passes ("canonicalize"), or as heir-opt flags
  # ("--mlir-to-bgv=min-slot-count=8192").
  heir.run(
      module,
      heir.Pass("annotate-module", backend="openfhe", scheme="bgv"),
      heir.Pass("mlir-to-bgv", min_slot_count=8192),
      "scheme-to-openfhe",
  )

  # Run code emitters, as with heir-translate. Options are keyword arguments.
  print(heir.translate(module, "emit-openfhe-pke-header"))
  print(heir.translate(module, "emit-openfhe-pke",
                       openfhe_include_type="install-relative"))

# Build IR directly with the generated op builders.
with heir.module() as m:
  i32 = types.i32()

  @func.FuncOp.from_py_func(i32)
  def conceal_add(x):
    y = arith.AddIOp(x, arith.ConstantOp(i32, 7))
    return secret.ConcealOp(types.secret(i32), y).result

  print(m)
```

`heir.translations()` lists the available code emitters.

## Development

The package is built with bazel:

- `//lib/Bindings/Python` builds the native code: a C API for HEIR
  (`HEIRCAPI.h`), a shared library holding all compiler code
  (`libHEIRPythonCAPI.so`), upstream's core extension (`_mlir.so`), and HEIR's
  extension (`_heir.so`).
- `//python:heir_pkg` assembles the importable package tree from the upstream
  MLIR Python sources, the generated dialect bindings, the native libraries, and
  the hand-written sources in `python/heir/`.
- `//python:heir` is a `py_library` for depending on the package from bazel.

Run the tests with

```bash
bazel test //python:heir_test
```

To use a locally built package outside of bazel, either install it (this runs
the bazel build):

```bash
pip install .
```

or install it in editable mode:

```bash
pip install -e .
```

An editable install imports the hand-written sources from `python/heir/`, so
edits to them take effect immediately, and finds the generated parts of the
package (and the `heir-opt` and `heir-translate` binaries) in `bazel-bin`. After
changing C++ code, update it with `bazel build //python:heir_pkg` (and
`//tools:heir-opt`, `//tools:heir-translate`). Alternatively, point `PYTHONPATH`
at the bazel output:

```bash
bazel build //python:heir_pkg
PYTHONPATH=$PWD/bazel-bin/python/heir_pkg python -c "import heir"
```

When adding a new dialect, add it to `HEIR_DIALECTS` in `python/BUILD` to
generate its Python op bindings.

The package does not include HEIR's Yosys/ABC-based passes and pipelines
(`--yosys-optimizer` and `--mlir-to-cggi`), neither in the Python bindings nor
in the bundled `heir-opt`, which is built with `--//:enable_yosys=0`.

<!-- mdformat global-off -->
