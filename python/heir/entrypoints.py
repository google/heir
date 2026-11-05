"""Console entrypoints for the heir-opt and heir-translate binaries bundled with
the wheel."""

import os
from pathlib import Path
import sys


def _run_binary(binary_name: str) -> None:
  """Locates and executes a bundled binary, replacing the current process."""
  binary_path = Path(__file__).parent / binary_name
  if not binary_path.exists():
    print(
        f"Error: {binary_name} not found in package at {binary_path}",
        file=sys.stderr,
    )
    sys.exit(1)
  os.execv(binary_path, [str(binary_path)] + sys.argv[1:])


def run_heir_opt() -> None:
  _run_binary("heir-opt")


def run_heir_translate() -> None:
  _run_binary("heir-translate")
