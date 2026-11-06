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
- The `heir-opt` and `heir-translate` command line tools (in the wheel only; see
  below).

The package requires Python 3.12+. Its native extensions are built against
CPython's stable ABI, so a single wheel per platform supports Python 3.12 and
all later versions.

## Example

Run from the root of a HEIR checkout:

```python
import heir
from heir import types
from heir.dialects import arith, func, secret

with heir.context():
  # Parse MLIR, or build it with the op builders (see below).
  module = heir.parse_file("tests/Examples/openfhe/ckks/conv_1d/conv_1d.mlir")

  # Run passes and pipelines, as with heir-opt. This is the pipeline of the
  # conv_1d test (heir_opt_flags in tests/Examples/openfhe/ckks/conv_1d/BUILD):
  #   heir-opt --annotate-module="backend=openfhe scheme=ckks" \
  #     --mlir-to-ckks=min-slot-count=32 --scheme-to-openfhe
  # Passes can be given as heir.Pass objects, as textual passes
  # ("scheme-to-openfhe"), or as heir-opt flags ("--mlir-to-ckks=...").
  heir.run(
      module,
      heir.Pass("annotate-module", backend="openfhe", scheme="ckks"),
      heir.Pass("mlir-to-ckks", min_slot_count=32),
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

To use the package outside of bazel, install it in editable mode:

1. Create and activate a virtual environment with Python 3.12+:
   `python3 -m venv venv && source venv/bin/activate`
1. Run `pip install -e .` from the repository root. This builds
   `//python:heir_pkg` with bazel, optimized like the published wheel.
1. Use it, e.g., run the example above, or `python python/test/heir_test.py`.

Edits to `python/heir/*.py` take effect immediately. The rest of the package
(the generated bindings and native libraries) is loaded from a copy of the
bazel-built package in `build/editable/heir`, so re-run `pip install -e .` after
changing C++ code.

`pip install .` instead builds and installs a copy of the whole package, like
the published wheel.

When adding a new dialect, add it to `HEIR_DIALECTS` in `python/BUILD` to
generate its Python op bindings.

The bundled `heir-opt` and `heir-translate` are thin launchers
(`//lib/Bindings/Python:heir-opt-launcher` and `:heir-translate-launcher`) that
run the tools from `libHEIRPythonCAPI.so`, rather than separate copies of the
compiler. Standalone builds of the tools for each platform are attached to
HEIR's [GitHub releases](https://github.com/google/heir/releases) instead (see
`HEIR_STANDALONE_BINARIES_DIR` in `setup.py`).

The package does not include HEIR's Yosys/ABC-based passes and pipelines
(`--yosys-optimizer` and `--mlir-to-cggi`), neither in the Python bindings nor
in the bundled tools.

<!-- mdformat global-off -->
