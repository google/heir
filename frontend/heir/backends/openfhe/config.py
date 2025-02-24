"""Configuration of OpenFHE backend."""

import dataclasses
import os

dataclass = dataclasses.dataclass


@dataclass(frozen=True)
class OpenFHEConfig:
  """Configuration for the OpenFHE backend.

  Attributes:
    include_dirs: The include paths for OpenFHE headers
    lib_dir: The directory containing libOPENFHEbinfhe.so, etc.
    link_libs: The libraries to link against (without lib prefix or .so suffix)
    include_type: The type of include paths to use during codegen. Options are:
      - "install-relative": use paths relative to the installed OpenFHE -
      "source-relative": relative to the openfhe development repository.
  """

  include_dirs: list[str]
  lib_dir: str
  link_libs: list[str]
  include_type: str = "install-relative"


DEFAULT_INSTALLED_OPENFHE_CONFIG = OpenFHEConfig(
    include_dirs=[
        "/usr/local/include/openfhe",
        "/usr/local/include/openfhe/binfhe",
        "/usr/local/include/openfhe/core",
        "/usr/local/include/openfhe/pke",
    ],
    lib_dir="/usr/local/lib",
    link_libs=[
        "OPENFHEbinfhe",
        "OPENFHEcore",
        "OPENFHEpke",
    ],
)


def from_os_env(debug=False) -> OpenFHEConfig:
  """Create an OpenFHEConfig from environment variables.

  Note, this is required for running tests under bazel, as the openfhe
  libraries,
  headers, and locations are not in the default locations.

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

  lib_dir = os.environ.get("OPENFHE_LIB_DIR", "")
  include_dirs = os.environ.get("OPENFHE_INCLUDE_DIR", "").split(":")
  link_libs = os.environ.get("OPENFHE_LINK_LIBS", "").split(":")

  # remove empty strings
  include_dirs = [dir for dir in include_dirs if dir]
  link_libs = [lib for lib in link_libs if lib]

  # Special case for bazel, RUNFILES_DIR is in OSS, TEST_SRCDIR is
  # for Google-internal testing.
  if "RUNFILES_DIR" in os.environ or "TEST_SRCDIR" in os.environ:
    path_base = os.getenv("RUNFILES_DIR", os.getenv("TEST_SRCDIR", ""))
    # bazel data dep on @openfhe//:core puts libcore.so in the
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

  return OpenFHEConfig(
      include_dirs=include_dirs
      or DEFAULT_INSTALLED_OPENFHE_CONFIG.include_dirs,
      lib_dir=lib_dir or DEFAULT_INSTALLED_OPENFHE_CONFIG.lib_dir,
      link_libs=link_libs or DEFAULT_INSTALLED_OPENFHE_CONFIG.link_libs,
      include_type=os.environ.get("OPENFHE_INCLUDE_TYPE", "install-relative"),
  )
