"""Console entrypoints for the heir-opt and heir-translate binaries bundled with
the wheel."""

import os
from pathlib import Path
import sys


def _find_binary(binary_name: str) -> Path:
  """Locates a bundled binary."""
  here = Path(__file__).resolve().parent
  candidates = [
      # A wheel bundles the binaries in the package.
      here / binary_name,
      # An editable install (`pip install -e .`) imports the package from the
      # source tree, and uses the binaries built by bazel.
      here.parents[1] / "bazel-bin" / "tools" / binary_name,
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
  """Executes a bundled binary, replacing the current process."""
  binary_path = _find_binary(binary_name)
  os.execv(binary_path, [str(binary_path)] + sys.argv[1:])


def run_heir_opt() -> None:
  _run_binary("heir-opt")


def run_heir_translate() -> None:
  _run_binary("heir-translate")
