"""Setup script for the heir package."""

from collections.abc import Generator
import contextlib
import os
import pathlib
import platform
import re
import shutil
from typing import Any

import setuptools
from setuptools.command import build_ext


Path = pathlib.Path
IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# The oldest supported Python version (keep in sync with `requires-python` in
# pyproject.toml). The native extensions are built against CPython's stable ABI
# (abi3) of this version, see the Py_LIMITED_API define in
# patches/nanobind_stable_abi.patch, so that a single wheel per platform
# supports this and all later Python versions.
MIN_PYTHON = (3, 12)
MIN_PYTHON_STR = "{}.{}".format(*MIN_PYTHON)
py_limited_api = True
options = {"bdist_wheel": {"py_limited_api": "cp{}{}".format(*MIN_PYTHON)}}


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


def _copy_file(src: Path, dst: Path, executable: bool) -> None:
  """Copies `src` to `dst`, replacing `dst` if it exists.

  Bazel outputs are read-only. Copying them with their permission bits (as
  shutil.copy2 does) would leave read-only files in the build directory, which
  the next build (pip reuses the build directory) could not overwrite.
  """
  if dst.exists() or dst.is_symlink():
    dst.unlink()
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(src, dst)
  os.chmod(dst, 0o755 if executable else 0o644)


def _copy_tree(src: Path, dst: Path) -> None:
  """Copies the files in the directory `src` into `dst` (see _copy_file)."""
  for path in sorted(src.rglob("*")):
    if path.is_dir():
      continue
    _copy_file(
        path, dst / path.relative_to(src), executable=path.suffix == ".so"
    )


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
      self.build_for_editable_install()
    else:
      for ext in self.extensions:
        self.bazel_build(ext)
    # explicitly call `bazel shutdown` for graceful exit
    self.spawn(["bazel", "shutdown"])

  def copy_extensions_to_source(self):
    """Copy generated extensions into the source tree.

    This is done in the ``bazel_build`` method, so it's not necessary to
    do again in the `build_ext` base class.
    """

  def build_for_editable_install(self) -> None:
    """Builds the package for an editable install (`pip install -e .`).

    An editable install imports `heir` from the source tree (python/heir). Its
    __init__.py finds the generated parts of the package (the upstream MLIR
    python sources, the generated dialect bindings, and the native libraries)
    in bazel-bin/python/heir_pkg, and the heir-opt/heir-translate entry points
    find the binaries in bazel-bin/tools. Build them with the default bazel
    configuration, so that a later `bazel build //python:heir_pkg` updates the
    editable install.
    """
    self.spawn([
        "bazel",
        "build",
        "//python:heir_pkg",
        "//tools:heir-opt",
        "//tools:heir-translate",
    ])

  def bazel_build(self, ext: BazelExtension) -> None:  # noqa: C901
    """Runs the bazel build to create the package."""
    temp_path = Path(self.build_temp)

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
        # Build the extensions against the oldest supported Python version's
        # headers (whichever Python runs this script): the extensions use the
        # stable ABI, so they work with all later versions.
        f"--@rules_python//python/config_settings:python_version={MIN_PYTHON_STR}",
        # The wheel doesn't ship Yosys's techmap files or the ABC binary, so
        # build heir-opt without the passes that need them.
        "--//:enable_yosys=0",
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
      # The tree is the complete package. Replace whatever an earlier build
      # left in libdir (pip reuses the build directory), so that stale files
      # can't end up in the wheel.
      if libdir.exists():
        shutil.rmtree(libdir)
      print(f"Copying {srcdir_path} to {libdir}")
      _copy_tree(srcdir_path, libdir)
      return

    libdir_path = libdir / ext.target_file
    print(f"Copying {srcdir_path} to {libdir_path}")
    _copy_file(srcdir_path, libdir_path, executable=ext.is_binary)


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
        # the native extensions, and the hand-written python sources. Must come
        # first, since building it replaces the package directory.
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
