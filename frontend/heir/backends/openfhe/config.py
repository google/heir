"""Configuration of OpenFHE backend."""

import dataclasses
import importlib.resources
import os
from heir.backends.util.common import get_repo_root, is_pip_installed

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


DEFAULT_INSTALLED_OPENFHE_CONFIG = OpenFHEConfig(
    include_dirs=[
        "/usr/local/include/openfhe",
        "/usr/local/include/openfhe/binfhe",
        "/usr/local/include/openfhe/core",
        "/usr/local/include/openfhe/pke",
    ],
    include_type="install-relative",
    lib_dir="/usr/local/lib",
    link_libs=[
        "openfhe",  # libopenfhe.so
    ],
)


def development_openfhe_config() -> OpenFHEConfig:
  repo_root = get_repo_root()
  if not repo_root:
    raise RuntimeError("Could not build development config. Did you run bazel?")

  return OpenFHEConfig(
      include_dirs=[
          str(repo_root / "external" / "openfhe"),
          str(
              repo_root / "external" / "openfhe" / "src" / "binfhe" / "include"
          ),
          str(repo_root / "external" / "openfhe" / "src" / "core" / "include"),
          str(repo_root / "external" / "openfhe" / "src" / "pke" / "include"),
          str(repo_root / "external" / "cereal" / "include"),
      ],
      include_type="source-relative",
      lib_dir=str(repo_root / "bazel-bin" / "external" / "openfhe"),
      link_libs=["openfhe"],
  )


def from_os_env(debug=False) -> OpenFHEConfig:
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
      the OpenFHEConfig
  """
  if debug:
    print("Env:")
    print(f"RUNFILES_DIR: {os.environ.get('RUNFILES_DIR', '')}")
    for k, v in os.environ.items():
      if "OPENFHE" in k:
        print(f"{k}: {v}")

  include_dirs = os.environ.get("OPENFHE_INCLUDE_DIR", "").split(":")
  include_type = os.environ.get("OPENFHE_INCLUDE_TYPE", "")
  lib_dir = os.environ.get("OPENFHE_LIB_DIR", "")
  link_libs = os.environ.get("OPENFHE_LINK_LIBS", "").split(":")

  # remove empty strings from lists
  include_dirs = [dir for dir in include_dirs if dir]
  link_libs = [lib for lib in link_libs if lib]

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

  for include_dir in include_dirs:
    if not os.path.exists(include_dir):
      print(
          f'Warning: OpenFHE include directory "{include_dir}" does not exist'
      )

  # If something has been found from the environment variables, return it
  if include_dirs:
    return OpenFHEConfig(
        include_dirs=include_dirs,
        lib_dir=lib_dir,
        link_libs=link_libs,
        include_type=include_type,
    )

  # if nothing is found, check the default installed config
  if debug:
    print(
        "HEIRpy Debug (OpenFHE Backend): No valid OpenFHE config found in"
        " environment variables, trying default install location."
    )
  if os.path.exists(DEFAULT_INSTALLED_OPENFHE_CONFIG.include_dirs[0]):
    return DEFAULT_INSTALLED_OPENFHE_CONFIG

  # if nothing is found still, check the development config
  if debug:
    print(
        "HEIRpy Debug (OpenFHE Backend): No valid OpenFHE config found in"
        " environment variables or default install location, trying"
        " development location."
    )
  return (
      development_openfhe_config()
  )  # will raise a RuntimeError if repo_root not found


def from_pip_installation() -> OpenFHEConfig:
  """
  Configure HEIR binaries from the expected pip installation structure.
  """
  if not is_pip_installed():
    raise RuntimeError("HEIR is not installed via pip.")

  package_path = importlib.resources.files("heir")
  return OpenFHEConfig(
      include_dirs=[
          str(package_path / "openfhe"),
          str(package_path / "openfhe" / "src" / "binfhe" / "include"),
          str(package_path / "openfhe" / "src" / "core" / "include"),
          str(package_path / "openfhe" / "src" / "pke" / "include"),
          str(package_path / "cereal" / "include"),
      ],
      include_type="source-relative",
      lib_dir=str(package_path),
      link_libs=["openfhe"],
  )
