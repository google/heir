"""Console entrypoints for the heir-opt and heir-translate executables bundled
with the package.

The bundled executables are thin launchers that run the tools from the shared
library holding all compiler code (`_mlir_libs/libHEIRPythonCAPI.so`). The
standalone tools are published as part of HEIR's GitHub releases.
"""

import os
from pathlib import Path
import sys

import heir


def _find_binary(binary_name: str) -> Path:
  """Locates a bundled executable."""
  # A wheel (or the bazel-built package) bundles the executables in the
  # package. An editable install (`pip install -e .`) imports the package from
  # the source tree, and finds them with the other generated parts of the
  # package (see __init__.py), which are on the package's search path.
  candidates = [Path(path) / binary_name for path in heir.__path__]
  for candidate in candidates:
    if candidate.is_file():
      return candidate
  print(
      f"Error: {binary_name} not found; looked in"
      f" {', '.join(str(c) for c in candidates)}",
      file=sys.stderr,
  )
  sys.exit(1)


def _run_binary(binary_name: str) -> None:
  """Executes a bundled executable, replacing the current process."""
  binary_path = _find_binary(binary_name)
  os.execv(binary_path, [str(binary_path)] + sys.argv[1:])


def run_heir_opt() -> None:
  _run_binary("heir-opt")


def run_heir_translate() -> None:
  _run_binary("heir-translate")
