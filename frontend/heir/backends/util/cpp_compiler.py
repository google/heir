"""A helper to find and run a C++ compiler."""

import os
import pathlib
import shutil
import subprocess
from typing import Callable, Optional

Path = pathlib.Path


# I would specify -stdlib here, but that would prevent the C compiler from discovering
# the includes automatically and searching for them when compiling. Probably best
# to let the C compiler try to automate it, rather than manually pass in stdlib paths.
DEFAULT_COMPILER_FLAGS: list[str] = ["-O3", "-fPIC", "-shared", "-std=c++17"]


def to_cpp_compiler_args(prefix, strs):
  return [f"{prefix}{s}" for s in strs] if strs else []


class CppCompilerBackend:
  """A helper to find and run a C++ compiler."""

  def __init__(self, path_to_compiler: Optional[str] = None):
    self.compiler_binary_path = path_to_compiler or self._find_cpp_compiler()

  def _find_cpp_compiler(self):
    compiler = os.environ.get("CXX", os.environ.get("CC"))
    if compiler:
      return compiler
    result = (
        shutil.which("clang++")
        or shutil.which("clang")
        or shutil.which("g++")
        or shutil.which("gcc")
    )
    if not result:
      raise ValueError(
          "Could not find a C++ compiler (clang++ or g++)."
          " Please install one or set the CXX"
          " environment variable to the path of the compiler."
      )
    return result

  def compile_to_shared_object(
      self,
      cpp_source_filepath: Path,
      shared_object_output_filepath: Path,
      compiler_flags: list[str] = DEFAULT_COMPILER_FLAGS,
      include_paths: Optional[list[str]] = None,
      linker_search_paths: Optional[list[str]] = None,
      link_libs: Optional[list[str]] = None,
      linker_args: Optional[list[str]] = None,
      abs_link_lib_paths: Optional[list[str | Path]] = None,
      arg_printer: Optional[Callable] = None,
  ):
    """Compile C++ source to a shared object file.

    Args:
        cpp_source: the C++ source code to compile
        shared_object_output_filepath: the path to the output .so file
        compiler_flags: the compiler flags to pass to the C++ compiler
        include_paths: include paths (-I) to pass to the C++ compiler
        linker_search_paths: linker search paths (-L) to pass to the C++ compiler
        link_libs: link libraries (-l) to pass to the C++ compiler
        linker_args: arguments to pass to the linker, via `-Wl,` prefix. The
          commas separating the values are added by this function.
        abs_link_lib_paths: absolute paths to libraries to link against.
    """
    # err if output filepath does not end with .so
    if shared_object_output_filepath.suffix != ".so":
      raise ValueError(
          "Expected shared object output filepath to end with .so, but got"
          f" {shared_object_output_filepath}"
      )

    include_args = to_cpp_compiler_args("-I", include_paths)
    linker_search_path_args = to_cpp_compiler_args("-L", linker_search_paths)
    link_lib_args = to_cpp_compiler_args("-l", link_libs)
    abs_link_lib_paths = abs_link_lib_paths or []

    if linker_args:
      linker_args = ["-Wl," + ",".join(str(x) for x in linker_args)]
    else:
      linker_args = []

    args = (
        [
            self.compiler_binary_path,
            cpp_source_filepath,
            "-o",
            shared_object_output_filepath,
        ]
        + abs_link_lib_paths
        + compiler_flags
        + include_args
        + linker_search_path_args
        + link_lib_args
        + linker_args
    )
    if arg_printer:
      arg_printer(args)

    completed_process = subprocess.run(
        args,
        text=True,
        capture_output=True,
    )
    if completed_process.returncode != 0:
      message = f"Error running {self.compiler_binary_path}. "
      message += "stderr was:\n\n" + completed_process.stderr + "\n\n"
      message += "options were:\n\n" + " ".join([str(x) for x in args])
      raise ValueError(message)
