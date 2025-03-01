"""Configuration of HEIR compiler toolchain."""

import dataclasses
import os
import pathlib
from pathlib import Path
import shutil

dataclass = dataclasses.dataclass


@dataclass(frozen=True)
class HEIRConfig:
  heir_opt_path: str | Path
  heir_translate_path: str | Path


def find_above(dirname: str) -> Path | None:
  path = pathlib.Path(__file__).resolve()
  matching = [p / dirname for p in path.parents if (p / dirname).exists()]
  return matching[-1] if matching else None


def get_repo_root() -> Path | None:
  default = find_above("bazel-bin")
  found = os.getenv("HEIR_REPO_ROOT_MARKER")
  return Path(found) if found else default


def development_heir_config() -> HEIRConfig:
  repo_root = get_repo_root()
  if not repo_root:
    raise RuntimeError("Could not build development config. Did you run bazel?")

  return HEIRConfig(
      heir_opt_path=repo_root / "tools" / "heir-opt",
      heir_translate_path=repo_root / "tools" / "heir-translate",
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
