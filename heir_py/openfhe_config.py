"""Configuration of OpenFHE backend."""

import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenFHEConfig:
    # The include paths for OpenFHE headers
    include_dirs: list[str]

    # The directory containing libOPENFHEbinfhe.so, etc.
    lib_dir: str

    # The libraries to link against (without lib prefix or .so suffix)
    link_libs: list[str]

    # The type of include paths to use during codegen. Options are:
    # - "install-relative": use paths relative to the installed OpenFHE
    # - "source-relative": relative to the openfhe development repository.
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

    Note, this is required for running tests under bazel, as the openfhe libraries,
    headers, and locations are not in the default locations.

    Environment variables and meanings:

    - OPENFHE_LIB_DIR: a string containing the directory containing the OpenFHE .so files.
    - OPENFHE_INCLUDE_DIR: a colon-separated string of directories containing OpenFHE headers.
    - OPENFHE_LINK_LIBS: a colon-separated string of libraries to link against
        (without `lib` or `.so`).
    - OPENFHE_INCLUDE_TYPE: a string indicating the include path type to use
        (see options on heir-translate --emit-openfhe).
    - RUNFILES_DIR: a directory prefix for all other paths provided, mainly for
        bazel runtime sandboxing.

    Args:
        debug: whether to print debug information

    Returns: the OpenFHEConfig
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

    # Special case for bazel
    if "RUNFILES_DIR" in os.environ:
        path_base = os.environ["RUNFILES_DIR"]
        # bazel data dep on @openfhe//:core puts libcore.so in the
        # $RUNFILES/openfhe dir
        lib_dir = os.path.join(path_base, lib_dir)
        # bazel data dep on @openfhe//:headers copies header files
        # to $RUNFILES_DIR/openfhe/src/...
        include_dirs = [os.path.join(path_base, dir) for dir in include_dirs]

    for dir in include_dirs:
        if not os.path.exists(dir):
            print(f"Warning: include directory {dir} does not exist")

    return OpenFHEConfig(
        include_dirs=include_dirs,
        lib_dir=lib_dir,
        link_libs=link_libs,
        include_type=os.environ.get("OPENFHE_INCLUDE_TYPE", "install-relative"),
    )
