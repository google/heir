"""Entrypoints for shimming heir-opt and heir-translate in the Python path"""

import os
from pathlib import Path
import sys


def _run_binary(binary_name):
  """Locate and execute a bundled binary."""
  # This file is at frontend/heir/entrypoints.py
  # The binaries are installed in the 'heir' package directory (frontend/heir/)
  binary_path = Path(__file__).parent / binary_name
  if not binary_path.exists():
    # Fallback for some environments where binaries might be in the parent
    binary_path = Path(__file__).parent.parent / binary_name

  if not binary_path.exists():
    print(
        f"Error: {binary_name} not found in package at {binary_path}",
        file=sys.stderr,
    )
    sys.exit(1)

  # os.execv replaces the current process with the binary
  os.execv(binary_path, [str(binary_path)] + sys.argv[1:])


def run_heir_opt():
  _run_binary("heir-opt")


def run_heir_translate():
  _run_binary("heir-translate")
