"""Configuration of OpenFHE backend."""

import dataclasses
import importlib.resources
import importlib.util
import os
import pathlib
import platform
import sys
import sysconfig
from typing import Optional
from heir.backends.util.common import get_repo_root, is_pip_installed
import traceback

Path = pathlib.Path

dataclass = dataclasses.dataclass


@dataclass(frozen=True)
class OpenFHEConfig:
  """Configuration for the OpenFHE backend.

  Attributes:
    include_dirs: The include paths for OpenFHE headers
    include_type: The type of include paths to use during codegen. Options are:
      - "install-relative": use paths relative to the installed OpenFHE
      - "source-relative": relative to the openfhe development repository.
    lib_dir: The directory containing shared libraries to link against
      (e.g., libopenfhe.so).
    link_libs: The libraries to link against (without lib prefix or .so suffix)
  """

  include_dirs: list[str]
  include_type: str
  lib_dir: str
  link_libs: list[str]


def get_default_installed_config() -> Optional[OpenFHEConfig]:
  prefixes = [Path("/usr/local"), Path("/usr")]
  for prefix in prefixes:
    include_base = prefix / "include" / "openfhe"
    if include_base.is_dir():
      lib_dir = None
      for lib_name in ["lib", "lib64"]:
        candidate_lib = prefix / lib_name
        if candidate_lib.is_dir():
          libs = _resolve_link_libs(candidate_lib)
          if libs != ["openfhe"]:
            lib_dir = candidate_lib
            break

      if not lib_dir:
        lib_dir = prefix / "lib"

      include_dirs = [
          str(include_base),
          str(include_base / "binfhe"),
          str(include_base / "core"),
          str(include_base / "pke"),
          str(prefix / "include"),
      ]
      include_dirs = [d for d in include_dirs if os.path.exists(d)]

      return OpenFHEConfig(
          include_dirs=include_dirs,
          include_type="install-relative",
          lib_dir=str(lib_dir),
          link_libs=_resolve_link_libs(lib_dir),
      )
  return None


def development_openfhe_config() -> OpenFHEConfig:
  repo_root = get_repo_root()
  if not repo_root:
    raise RuntimeError("Could not build development config. Did you run bazel?")

  return OpenFHEConfig(
      include_dirs=[
          str(repo_root / "bazel-heir" / "external" / "openfhe+"),
          str(
              repo_root
              / "bazel-heir"
              / "external"
              / "openfhe+"
              / "src"
              / "binfhe"
              / "include"
          ),
          str(
              repo_root
              / "bazel-heir"
              / "external"
              / "openfhe+"
              / "src"
              / "core"
              / "include"
          ),
          str(
              repo_root
              / "bazel-heir"
              / "external"
              / "openfhe+"
              / "src"
              / "pke"
              / "include"
          ),
          str(repo_root / "bazel-heir" / "external" / "cereal+" / "include"),
      ],
      include_type="source-relative",
      lib_dir=str(repo_root / "bazel-bin" / "external" / "openfhe+"),
      link_libs=["openfhe"],
  )


def _resolve_link_libs(lib_dir: os.PathLike | str) -> list[str]:
  lib_path = Path(lib_dir)
  if not lib_path.is_dir():
    return ["openfhe"]

  core_lib = None
  pke_lib = None
  binfhe_lib = None

  for p in lib_path.iterdir():
    if not p.is_file():
      continue
    name = p.name.lower()
    if not (".so" in name or ".dylib" in name):
      continue

    if "openfhe" in name:
      if "core" in name:
        core_lib = p
      elif "pke" in name:
        pke_lib = p
      elif "binfhe" in name:
        binfhe_lib = p

  if core_lib and pke_lib and binfhe_lib:
    return [
        str(core_lib.resolve()),
        str(pke_lib.resolve()),
        str(binfhe_lib.resolve()),
    ]

  link_libs = []
  for p in lib_path.iterdir():
    if not p.is_file():
      continue
    name = p.name.lower()
    if not (".so" in name or ".dylib" in name):
      continue
    if "openfhe" in name and not any(
        x in name for x in ["core", "pke", "binfhe"]
    ):
      link_libs.append(str(p.resolve()))

  if link_libs:
    return link_libs
  return ["openfhe"]


def from_os_env(debug: bool = False) -> Optional[OpenFHEConfig]:
  """Create an OpenFHEConfig from environment variables.

  Note, this is required for running tests under bazel, as the openfhe
  libraries, headers, and locations are not in the default locations.

  Environment variables and meanings:

  - OPENFHE_LIB_DIR: a string containing the directory containing the OpenFHE
  .so files.
  - OPENFHE_INCLUDE_DIR: a colon-separated string of directories containing
  OpenFHE headers.
  - OPENFHE_LINK_LIBS: a colon-separated string of libraries to link against
      (without `lib` or `.so`).
  - OPENFHE_INCLUDE_TYPE: a string indicating the include path type to use
      (see options on heir-translate --emit-openfhe).
  - RUNFILES_DIR: a directory prefix for all other paths provided, mainly for
      bazel runtime sandboxing.

  Args:
      debug: whether to print debug information

  Returns:
      the OpenFHEConfig or None if no OPENFHE environment variables are set.
  """
  trigger_keys = {"OPENFHE_INCLUDE_DIR", "OPENFHE_LIB_DIR", "OPENFHE_LINK_LIBS"}
  if not any(os.environ.get(k) for k in trigger_keys):
    return None

  env_keys = [k for k in os.environ.keys() if "OPENFHE" in k]

  if debug:
    print("Env:")
    print(f"RUNFILES_DIR: {os.environ.get('RUNFILES_DIR', '')}")
    for k in env_keys:
      print(f"{k}: {os.environ.get(k)}")

  inc_str = os.environ.get("OPENFHE_INCLUDE_DIR", "")
  lib_dir = os.environ.get("OPENFHE_LIB_DIR", "")

  if not inc_str or not lib_dir:
    raise ValueError(
        "Both OPENFHE_INCLUDE_DIR and OPENFHE_LIB_DIR must be set when"
        f" overriding OpenFHE configuration. Found keys: {env_keys}"
    )

  include_dirs = inc_str.split(":")
  include_type = os.environ.get("OPENFHE_INCLUDE_TYPE", "install-relative")

  # Special case for bazel, RUNFILES_DIR is in OSS, TEST_SRCDIR is
  # for Google-internal testing.
  if "RUNFILES_DIR" in os.environ or "TEST_SRCDIR" in os.environ:
    path_base = os.getenv("RUNFILES_DIR", os.getenv("TEST_SRCDIR", ""))
    # bazel data dep on @openfhe//:core_shared puts libopenfhe.so in the
    # $RUNFILES/openfhe dir
    lib_dir = os.path.join(path_base, lib_dir)
    # bazel data dep on @openfhe//:headers copies header files
    # to $RUNFILES_DIR/openfhe/src/...
    include_dirs = [os.path.join(path_base, dir) for dir in include_dirs]

  link_libs_str = os.environ.get("OPENFHE_LINK_LIBS", "")
  if link_libs_str:
    link_libs = link_libs_str.split(":")
  else:
    link_libs = _resolve_link_libs(Path(lib_dir))

  extra_include_dirs = []
  for d in include_dirs:
    p = Path(d)
    if p.name == "openfhe":
      parent_dir = str(p.parent)
      if (
          parent_dir not in include_dirs
          and parent_dir not in extra_include_dirs
      ):
        extra_include_dirs.append(parent_dir)
  include_dirs.extend(extra_include_dirs)

  # remove empty strings from lists
  include_dirs = [dir for dir in include_dirs if dir]
  link_libs = [lib for lib in link_libs if lib]

  if not include_dirs or not link_libs:
    raise ValueError(
        "Invalid OpenFHE configuration: include_dirs or link_libs resolved to"
        f" empty. Found keys: {env_keys}"
    )

  for include_dir in include_dirs:
    if not os.path.exists(include_dir):
      print(
          f'Warning: OpenFHE include directory "{include_dir}" does not exist'
      )

  return OpenFHEConfig(
      include_dirs=include_dirs,
      lib_dir=lib_dir,
      link_libs=link_libs,
      include_type=include_type,
  )


def resolve_config(debug: bool = False) -> OpenFHEConfig:
  """Resolve OpenFHEConfig in cascading order of preference."""
  debug_active = debug or os.environ.get("OPENFHE_DEBUG") == "1"

  if debug_active:
    print("HEIRpy Debug (config): Starting resolve_config", file=sys.stderr)

  if debug_active:
    print("HEIRpy Debug (config): Attempting from_os_env()", file=sys.stderr)
  os_env_config = from_os_env(debug=debug_active)
  if os_env_config is not None:
    if debug_active:
      print(
          "HEIRpy Debug (config): Successfully resolved from_os_env():"
          f" {os_env_config}",
          file=sys.stderr,
      )
    return os_env_config
  else:
    if debug_active:
      print(
          "HEIRpy Debug (config): from_os_env() returned None", file=sys.stderr
      )

  if debug_active:
    print(
        "HEIRpy Debug (config): Attempting get_default_installed_config()",
        file=sys.stderr,
    )
  default_config = get_default_installed_config()
  if default_config is not None:
    if debug_active:
      print(
          "HEIRpy Debug (config): Successfully resolved"
          " get_default_installed_config():"
          f" {default_config}",
          file=sys.stderr,
      )
    return default_config
  else:
    if debug_active:
      print(
          "HEIRpy Debug (config): get_default_installed_config() returned None",
          file=sys.stderr,
      )

  if debug_active:
    print(
        "HEIRpy Debug (config): Critical fallthrough to"
        " development_openfhe_config()",
        file=sys.stderr,
    )
  return development_openfhe_config()
