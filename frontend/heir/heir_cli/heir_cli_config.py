"""Configuration of HEIR compiler toolchain."""

import dataclasses
import os
import shutil
import pathlib

from pathlib import Path

dataclass = dataclasses.dataclass


@dataclass(frozen=True)
class HEIRConfig:
  heir_opt_path: str | Path
  heir_translate_path: str | Path


def find_above(dirname: str):
  path = pathlib.Path(__file__).resolve()
  matching = [p / dirname for p in path.parents if (p / dirname).exists()]
  return matching[-1]


repo_root = find_above("bazel-bin")

DEVELOPMENT_HEIR_CONFIG = HEIRConfig(
    heir_opt_path=repo_root / "tools/heir-opt",
    heir_translate_path=repo_root / "tools/heir-translate",
)

# TODO (#1326): Add a config that automagically downloads the nightlies


def from_os_env() -> HEIRConfig:
  """Create a HEIRConfig from environment variables.

  Note, this is required for running tests under bazel, as the locations
  of the various binaries are determined by bazel.

  The order of preference is:

  1. Environment variable HEIR_OPT_PATH or HEIR_TRANSLATE_PATH
  2. The path to the heir-opt or heir-translate binary on the PATH
  3. The default development configuration (relative to the project root, in
     bazel-bin)

  Returns:
    The HEIRConfig
  """
  which_heir_opt = shutil.which("heir-opt")
  which_heir_translate = shutil.which("heir-translate")
  resolved_heir_opt_path = os.environ.get(
      "HEIR_OPT_PATH",
      which_heir_opt or DEVELOPMENT_HEIR_CONFIG.heir_opt_path,
  )
  resolved_heir_translate_path = os.environ.get(
      "HEIR_TRANSLATE_PATH",
      which_heir_translate or DEVELOPMENT_HEIR_CONFIG.heir_translate_path,
  )

  return HEIRConfig(
      heir_opt_path=resolved_heir_opt_path,
      heir_translate_path=resolved_heir_translate_path,
  )
