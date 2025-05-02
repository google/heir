"""Helper functions for pybind11.

We compile pybind11 bindings with a C compiler, and from the
pybind11 docs, the main way to do this is

$ c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes)
example.cpp -o example$(python3-config --extension-suffix)

This file provides helper functions that implement the two shell subprocess
steps

- pybind11_includes: $(python3 -m pybind11 --includes)
- pyconfig_ext_suffix: $(python3-config --extension-suffix)

To support this, two functions quote and get_include are forked from pybind11.
because Google's internal version of pybind doesn't provide access to the python
package.
"""

import os
import re
import sys
import sysconfig


# This is the conditional used for os.path being posixpath
if "posix" in sys.builtin_module_names:
  from shlex import quote
elif "nt" in sys.builtin_module_names:
  # See https://github.com/mesonbuild/meson/blob/db22551ed9d2dd7889abea01cc1c7bba02bf1c75/mesonbuild/utils/universal.py#L1092-L1121
  # and the original documents:
  # https://docs.microsoft.com/en-us/cpp/c-language/parsing-c-command-line-arguments and
  # https://blogs.msdn.microsoft.com/twistylittlepassagesallalike/2011/04/23/everyone-quotes-command-line-arguments-the-wrong-way/
  UNSAFE = re.compile("[ \t\n\r]")

  def quote(s: str) -> str:
    if s and not UNSAFE.search(s):
      return s

    # Paths cannot contain a '"' on Windows, so we don't need to worry
    # about nuanced counting here.
    return f'"{s}\\"' if s.endswith("\\") else f'"{s}"'

else:

  def quote(s: str) -> str:
    return s


# Forked from pybind11 because the only implementation prints the values
# immediately.
def pybind11_includes() -> list[str]:
  """Return the include directories for pybind11.

  Returns:
    A list of include directories for pybind11.
  """
  dirs = [
      sysconfig.get_path("include"),
      sysconfig.get_path("platinclude"),
  ]
  try:
    from pybind11.commands import get_include

    dirs.append(get_include())
  except ImportError:
    pybind11_include = os.environ.get("PYBIND11_INCLUDE_PATH", "")
    if not pybind11_include:
      raise ValueError(
          "PYBIND11_INCLUDE_PATH is not set and pybind11 is not pip installed."
      )
    if not os.path.isdir(pybind11_include):
      print(
          f"Warning: PYBIND11_INCLUDE_PATH {pybind11_include} is not a"
          " directory."
      )
    dirs.append(pybind11_include)

  # Make unique but preserve order
  unique_dirs = []
  for d in dirs:
    if d and d not in unique_dirs:
      unique_dirs.append(d)

  return [quote(d) for d in unique_dirs]


def pybind11_libs() -> str:
  """Return the system directories for pybind11.

  Returns:
    A list of directories for pybind11.
  """
  dirs = [sysconfig.get_config_var("LIBDIR")]
  py_lib_dir = os.path.dirname(sysconfig.get_path("stdlib"))
  if py_lib_dir:
    dirs.append(py_lib_dir)
  return [quote(d) for d in dirs]


def python_link_libs() -> list[str]:
  """Return the strs for the linker to include the python lib.

  For whatever reason, this is required on macos but not other platforms.

  Returns:
    A list of str of the python lib for the linker to include.
  """
  if sys.platform == "darwin":
    return [f"python{sysconfig.get_python_version()}"]
  else:
    return []


def pyconfig_ext_suffix() -> str:
  return sysconfig.get_config_var("EXT_SUFFIX")
