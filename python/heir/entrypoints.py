"""Console entrypoints for the heir-opt and heir-translate executables bundled
with the package.

The bundled executables are thin launchers that run the tools from the shared
library holding all compiler code (`_mlir_libs/libHEIRPythonCAPI.so`). The
standalone tools are published as part of HEIR's GitHub releases.
"""

import os
from pathlib import Path
import sys


def _find_binary(binary_name: str) -> Path:
  """Locates a bundled executable."""
  here = Path(__file__).resolve().parent
  candidates = [
      # A wheel (or the bazel-built package) bundles the executables in the
      # package.
      here / binary_name,
      # An editable install (`pip install -e .`) imports the package from the
      # source tree, and uses the bazel-built package (see __init__.py).
      here.parents[1]
      / "bazel-bin"
      / "python"
      / "heir_pkg"
      / "heir"
      / binary_name,
  ]
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
