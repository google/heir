"""Configuration of HEIR compiler toolchain."""

import dataclasses
import os
import pathlib
from pathlib import Path
import shutil
from heir.backends.util.common import get_repo_root

dataclass = dataclasses.dataclass


@dataclass(frozen=True)
class HEIRConfig:
  heir_opt_path: str | Path
  heir_translate_path: str | Path


def development_heir_config() -> HEIRConfig:
  repo_root = get_repo_root()
  if not repo_root:
    raise RuntimeError("Could not build development config. Did you run bazel?")

  return HEIRConfig(
      heir_opt_path=repo_root / "bazel-bin" / "tools" / "heir-opt",
      heir_translate_path=repo_root / "bazel-bin" / "tools" / "heir-translate",
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
      which_heir_opt or development_heir_config().heir_opt_path,
  )
  resolved_heir_translate_path = os.environ.get(
      "HEIR_TRANSLATE_PATH",
      which_heir_translate or development_heir_config().heir_translate_path,
  )

  return HEIRConfig(
      heir_opt_path=resolved_heir_opt_path,
      heir_translate_path=resolved_heir_translate_path,
  )
