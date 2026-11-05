"""Setup script for the heir package."""

from collections.abc import Generator
import contextlib
import os
import pathlib
import platform
import re
import shutil
import stat
import sys
from typing import Any

import setuptools
from setuptools.command import build_ext


Path = pathlib.Path
IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# The native extensions are built against CPython's stable ABI for 3.12+ (see
# patches/nanobind_stable_abi.patch), so a single abi3 wheel per platform works
# for all supported Python versions.
py_limited_api = True
options = {"bdist_wheel": {"py_limited_api": "cp312"}}


def is_cibuildwheel() -> bool:
  return os.getenv("CIBUILDWHEEL") is not None


@contextlib.contextmanager
def _maybe_patch_toolchains() -> Generator[None, None, None]:
  """Patch rules_python toolchains to ignore root user error

  when run in a Docker container on Linux in cibuildwheel.
  """

  def fmt_toolchain_args(matchobj):
    suffix = "ignore_root_user_error = True"
    callargs = matchobj.group(1)
    # toolchain def is broken over multiple lines
    if callargs.endswith("\n"):
      callargs = callargs + "    " + suffix + ",\n"
    # toolchain def is on one line.
    else:
      callargs = callargs + ", " + suffix
    return "python.toolchain(" + callargs + ")"

  CIBW_LINUX = is_cibuildwheel() and IS_LINUX
  module_bazel = Path("MODULE.bazel")
  content: str = module_bazel.read_text()
  try:
    if CIBW_LINUX:
      module_bazel.write_text(
          re.sub(
              r"python.toolchain\(([\w\"\s,.=]*)\)",
              fmt_toolchain_args,
              content,
          )
      )
    yield
  finally:
    if CIBW_LINUX:
      module_bazel.write_text(content)


class BazelExtension(setuptools.Extension):
  """A C/C++ extension that is defined as a Bazel BUILD target."""

  def __init__(
      self,
      name: str,
      bazel_target: str,
      generated_so_file: Path,
      target_file: str,
      is_binary: bool = False,
      is_tree: bool = False,
      aggressive_strip: bool = False,
      **kwargs: Any,
  ):
    super().__init__(name=name, sources=[], **kwargs)

    self.bazel_target = bazel_target
    # A tuple of strings representing path components
    # like ("path", "to", "file.so")
    self.generated_so_file = generated_so_file
    self.target_file = target_file
    stripped_target = bazel_target.split("//")[-1]
    self.relpath, self.target_name = stripped_target.split(":")
    self.is_binary = is_binary

    # Whether generated_so_file is a directory whose contents should be merged
    # into the package directory (target_file is then ignored).
    self.is_tree = is_tree

    # Determines whether to `strip-all` when building a target for PyPI.
    # Critically, this should only be used for cc_binary targets that are run as
    # a subprocess, not cc_library targets that are loaded into Python.
    self.aggressive_strip = aggressive_strip


class BuildBazelExtension(build_ext.build_ext):
  """A command that runs Bazel to build a C/C++ extension."""

  def run(self):
    if self.inplace:
      # This corresponds to pip install --editable. The package consists mostly
      # of bazel-generated files (MLIR python sources, generated dialect
      # bindings, and native extensions), so an editable install of the source
      # tree would not be importable.
      raise RuntimeError(
          "Editable installs are not supported. Either run `pip install .`, or"
          " run `bazel build //python:heir_pkg` and add"
          " `bazel-bin/python/heir_pkg` to PYTHONPATH."
      )

    for ext in self.extensions:
      self.bazel_build(ext)
    # explicitly call `bazel shutdown` for graceful exit
    self.spawn(["bazel", "shutdown"])

  def copy_extensions_to_source(self):
    """Copy generated extensions into the source tree.

    This is done in the ``bazel_build`` method, so it's not necessary to
    do again in the `build_ext` base class.
    """

  def bazel_build(self, ext: BazelExtension) -> None:  # noqa: C901
    """Runs the bazel build to create the package."""
    temp_path = Path(self.build_temp)

    # We round to the minor version, which makes rules_python
    # look up the latest available patch version internally.
    python_version = "{}.{}".format(*sys.version_info[:2])

    bazel_argv = [
        "bazel",
        "build",
        ext.bazel_target,
        # make output suitable for CI
        "--curses=no",
        "--ui_event_filters=ERROR",
        f"--symlink_prefix={temp_path / 'bazel-'}",
        "--compilation_mode=opt",
        "--strip=always",
        f"--cxxopt={'/std:c++20' if IS_WINDOWS else '-std=c++20'}",
        f"--@rules_python//python/config_settings:python_version={python_version}",
    ]

    if ext.aggressive_strip:
      if IS_LINUX:
        bazel_argv.append("--stripopt=--strip-all")
      elif IS_MAC:
        bazel_argv.append("--stripopt=-S")

    if IS_WINDOWS:
      # Link with python*.lib.
      for library_dir in self.library_dirs:
        bazel_argv.append("--linkopt=/LIBPATH:" + library_dir)
    elif IS_MAC:
      # C++17 needs macOS 10.14 at minimum
      bazel_argv.append("--macos_minimum_os=10.15")

      # Cross-compilation support: detect target arch from ARCHFLAGS (set by cibuildwheel).
      archflags = os.environ.get("ARCHFLAGS", "")
      if "x86_64" in archflags:
        target_arch = "x86_64"
      elif "arm64" in archflags:
        target_arch = "arm64"
      else:
        target_arch = platform.machine()
      bazel_argv.append(
          f"--platforms=@apple_support//platforms:darwin_{target_arch}"
      )

    with _maybe_patch_toolchains():
      self.spawn(bazel_argv)

    # copy the Bazel build artifacts into setuptools' libdir,
    # from where the wheel is built.
    print("\n\nCopying Bazel build artifacts to setuptools libdir\n\n")
    srcdir = temp_path / "bazel-bin"
    libdir = Path(self.build_lib) / "heir"

    # map from srcdir-relative paths to libdir-relative paths
    srcdir_path = srcdir / ext.generated_so_file
    if ext.is_tree:
      print(f"Copying {srcdir_path} to {libdir}")
      shutil.copytree(
          src=srcdir_path, dst=libdir, symlinks=False, dirs_exist_ok=True
      )
      return

    libdir_path = libdir / ext.target_file
    print(f"Copying {srcdir_path} to {libdir_path}")
    shutil.copyfile(srcdir_path, libdir_path)

    # run chmod +x on is_binary = True
    if ext.is_binary:
      # set executable bit on the target file
      target_path = libdir / ext.target_file
      print(f"Setting executable bit on {target_path}")
      # chmod 775 the file
      os.chmod(
          target_path,
          stat.S_IRUSR
          | stat.S_IWUSR
          | stat.S_IXUSR
          | stat.S_IRGRP
          | stat.S_IWGRP
          | stat.S_IXGRP
          | stat.S_IROTH
          | stat.S_IXOTH,
      )


setuptools.setup(
    cmdclass={
        "build_ext": BuildBazelExtension,
    },
    package_data={
        "heir": [
            "py.typed",
            "*.pyi",
        ]
    },
    ext_modules=[
        # The python package: MLIR python bindings, generated dialect bindings,
        # the native extensions, and the hand-written python sources.
        BazelExtension(
            name="heir._mlir_libs._heir",
            bazel_target="//python:heir_pkg",
            generated_so_file=Path("python") / "heir_pkg" / "heir",
            target_file="",
            py_limited_api=py_limited_api,
            is_tree=True,
        ),
        BazelExtension(
            name="heir._heir_opt",
            bazel_target="//tools:heir-opt.stripped",
            generated_so_file=Path("tools") / "heir-opt.stripped",
            target_file="heir-opt",
            py_limited_api=py_limited_api,
            is_binary=True,
            aggressive_strip=is_cibuildwheel(),
        ),
        BazelExtension(
            name="heir._heir_translate",
            bazel_target="//tools:heir-translate.stripped",
            generated_so_file=Path("tools") / "heir-translate.stripped",
            target_file="heir-translate",
            py_limited_api=py_limited_api,
            is_binary=True,
            aggressive_strip=is_cibuildwheel(),
        ),
    ],
    options=options,
)
