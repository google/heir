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


# An editable install (`pip install -e .`) imports the hand-written python
# sources from python/heir, and the rest of the package from a copy of the
# bazel-built package in this directory (see python/heir/__init__.py).
EDITABLE_GENERATED_DIR = Path(__file__).resolve().parent / "build" / "editable"

# The release workflows publish standalone heir-opt and heir-translate binaries
# for each platform; the wheel only bundles thin launchers, which run the tools
# from the shared library holding all compiler code. If this environment
# variable is set, also build the standalone binaries into the directory it
# names, with the wheel's configuration (so that most of the build is shared).
STANDALONE_BINARIES_DIR_ENV = "HEIR_STANDALONE_BINARIES_DIR"


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


_NATIVE_MAGIC_NUMBERS = (
    b"\x7fELF",
    b"\xcf\xfa\xed\xfe",  # Mach-O (64 bit)
    b"\xca\xfe\xba\xbe",  # Mach-O (universal)
)


def _is_native(path: Path) -> bool:
  """Whether `path` is an executable or a shared library."""
  with path.open("rb") as f:
    return f.read(4) in _NATIVE_MAGIC_NUMBERS


def _copy_tree(src: Path, dst: Path) -> None:
  """Copies the files in the directory `src` into `dst` (see _copy_file).

  Skips __pycache__ directories, which python creates in the bazel-built tree
  when something imports the package from there directly.
  """
  for path in sorted(src.rglob("*")):
    relative_path = path.relative_to(src)
    if path.is_dir() or "__pycache__" in relative_path.parts:
      continue
    _copy_file(path, dst / relative_path, executable=_is_native(path))


class BazelExtension(setuptools.Extension):
  """The python package, as a directory tree built by Bazel."""

  def __init__(
      self,
      name: str,
      bazel_target: str,
      generated_tree: Path,
      **kwargs: Any,
  ):
    super().__init__(name=name, sources=[], **kwargs)
    self.bazel_target = bazel_target
    # The directory under bazel-bin that holds the package.
    self.generated_tree = generated_tree


class BuildBazelExtension(build_ext.build_ext):
  """A command that runs Bazel to build the python package."""

  def run(self):
    for ext in self.extensions:
      self.bazel_build(ext)
    standalone_binaries_dir = os.environ.get(STANDALONE_BINARIES_DIR_ENV)
    if standalone_binaries_dir and not self.inplace:
      self.build_standalone_binaries(Path(standalone_binaries_dir))
    # explicitly call `bazel shutdown` for graceful exit
    self.spawn(["bazel", "shutdown"])

  def copy_extensions_to_source(self):
    """Copy generated extensions into the source tree.

    This is done in the ``bazel_build`` method, so it's not necessary to
    do again in the `build_ext` base class.
    """

  def bazel_argv(
      self, *targets: str, strip_binaries: bool = False
  ) -> list[str]:
    """Returns the command line that builds `targets` for the wheel."""
    temp_path = Path(self.build_temp)

    bazel_argv = [
        "bazel",
        "build",
        *targets,
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

    # Fully strip the *.stripped outputs of cc_binary targets. Only use this
    # for executables, not for shared libraries loaded into Python.
    if strip_binaries:
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
    return bazel_argv

  def bazel_build(self, ext: BazelExtension) -> None:
    """Builds the package, and copies it to where it is installed from.

    The wheel is built from setuptools' libdir. An editable install (`pip
    install -e .`) imports the hand-written sources from the source tree
    instead, and the rest of the package from a copy in EDITABLE_GENERATED_DIR
    (see python/heir/__init__.py), so that it uses the wheel's optimized build,
    and later bazel commands (which may change what bazel-bin points to, or
    delete files in it) don't break it.
    """
    with _maybe_patch_toolchains():
      self.spawn(self.bazel_argv(ext.bazel_target))

    srcdir = Path(self.build_temp) / "bazel-bin" / ext.generated_tree
    if self.inplace:
      libdir = EDITABLE_GENERATED_DIR / "heir"
    else:
      libdir = Path(self.build_lib) / "heir"
    # The tree is the complete package. Replace whatever an earlier build left
    # in libdir (pip reuses the build directory), so that stale files can't end
    # up in the package.
    if libdir.exists():
      shutil.rmtree(libdir)
    print(f"Copying {srcdir} to {libdir}")
    _copy_tree(srcdir, libdir)

  def build_standalone_binaries(self, outdir: Path) -> None:
    """Builds the standalone heir-opt and heir-translate into `outdir`."""
    names = ["heir-opt", "heir-translate"]
    with _maybe_patch_toolchains():
      self.spawn(
          self.bazel_argv(
              *[f"//tools:{name}.stripped" for name in names],
              strip_binaries=True,
          )
      )

    srcdir = Path(self.build_temp) / "bazel-bin" / "tools"
    for name in names:
      print(f"Copying {srcdir / name}.stripped to {outdir / name}")
      _copy_file(srcdir / f"{name}.stripped", outdir / name, executable=True)


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
        # The python package: the MLIR python bindings, the generated dialect
        # bindings, the native extensions and libraries, the heir-opt and
        # heir-translate launchers, and the hand-written python sources.
        BazelExtension(
            name="heir._mlir_libs._heir",
            bazel_target="//python:heir_pkg",
            generated_tree=Path("python") / "heir_pkg" / "heir",
            py_limited_api=py_limited_api,
        ),
    ],
    options=options,
)
