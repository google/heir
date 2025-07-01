"""Configuration of HEIR compiler toolchain."""

import dataclasses
import os
import importlib.resources
import pathlib
from pathlib import Path
import shutil
from heir.backends.util.common import get_repo_root, is_pip_installed


dataclass = dataclasses.dataclass


@dataclass(frozen=True)
class HEIRConfig:
  heir_opt_path: str | Path
  heir_translate_path: str | Path
  techmap_dir_path: str | Path  # optional
  abc_path: str | Path  # optional


def development_heir_config() -> HEIRConfig:
  repo_root = get_repo_root()
  if not repo_root:
    raise RuntimeError("Could not build development config. Did you run bazel?")

  techmap_dir_path = (
      repo_root
      / "bazel-bin"
      / "tools"
      / "heir-opt.runfiles"
      / "_main"
      / "lib"
      / "Transforms"
      / "YosysOptimizer"
      / "yosys"
  )
  if not techmap_dir_path.exists():
    techmap_dir_path = ""

  abc_path = (
      repo_root
      / "bazel-bin"
      / "tools"
      / "heir-opt.runfiles"
      / "edu_berkeley_abc"
      / "abc"
  )
  if not abc_path.exists():
    abc_path = ""

  return HEIRConfig(
      heir_opt_path=repo_root / "bazel-bin" / "tools" / "heir-opt",
      heir_translate_path=repo_root / "bazel-bin" / "tools" / "heir-translate",
      techmap_dir_path=techmap_dir_path,
      abc_path=abc_path,
  )


def from_os_env() -> HEIRConfig:
  """Create a HEIRConfig from environment variables.

  Note, this is required for running tests under bazel, as the locations
  of the various binaries are determined by bazel.

  The order of preference is:

  1. Environment variables HEIR_OPT_PATH, HEIR_TRANSLATE_PATH, HEIR_ABC_BINARY,
     or HEIR_YOSYS_SCRIPTS_DIR
  2. The path to the heir-opt or heir-translate binary on the PATH
  3. The default development configuration (relative to the project root, in
     bazel-bin)

  Returns:
    The HEIRConfig
  """
  which_heir_opt = shutil.which("heir-opt")
  which_heir_translate = shutil.which("heir-translate")
  which_abc = shutil.which("abc")
  resolved_heir_opt_path = os.environ.get(
      "HEIR_OPT_PATH",
      which_heir_opt or development_heir_config().heir_opt_path,
  )
  resolved_heir_translate_path = os.environ.get(
      "HEIR_TRANSLATE_PATH",
      which_heir_translate or development_heir_config().heir_translate_path,
  )
  resolved_abc_path = os.environ.get(
      "HEIR_ABC_BINARY",
      which_abc or development_heir_config().abc_path,
  )
  resolved_techmap_dir_path = os.environ.get(
      "HEIR_YOSYS_SCRIPTS_DIR",
      development_heir_config().techmap_dir_path,
  )

  return HEIRConfig(
      heir_opt_path=resolved_heir_opt_path,
      heir_translate_path=resolved_heir_translate_path,
      techmap_dir_path=resolved_techmap_dir_path,
      abc_path=resolved_abc_path,
  )


def from_pip_installation() -> HEIRConfig:
  """
  Configure HEIR binaries from the expected pip installation structure.
  """
  if not is_pip_installed():
    raise RuntimeError("HEIR is not installed via pip.")

  package_path = importlib.resources.files("heir")
  return HEIRConfig(
      heir_opt_path=package_path / "heir-opt",
      heir_translate_path=package_path / "heir-translate",
      # These paths are configured in setup.py
      techmap_dir_path=package_path / "techmaps",
      abc_path=package_path / "abc",
  )
