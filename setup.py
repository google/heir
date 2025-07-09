"""Setup script for the heir package."""

from collections.abc import Generator
import contextlib
import fnmatch
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

# hardcoded SABI-related options. Requires that each Python interpreter
# (hermetic or not) participating is of the same major-minor version.
py_limited_api = sys.version_info >= (3, 11)
options = {"bdist_wheel": {"py_limited_api": "cp311"}} if py_limited_api else {}


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


# A hack to tell shutils.copytree to only include files of a certain kind
def include_patterns(patterns):
  """Function that can be used as shutil.copytree() ignore parameter.

  Copies only files matching the given patterns.
  """

  def _ignore_patterns(path, names):
    # Return all names that don't match any pattern
    keep = set()
    for pattern in patterns:
      keep |= set(fnmatch.filter(names, pattern))

    # if it's a directory, keep it to ensure recursive search
    for name in names:
      if (Path(path) / name).resolve().is_dir():
        keep.add(name)

    return set(names) - keep

  return _ignore_patterns


class BazelExtension(setuptools.Extension):
  """A C/C++ extension that is defined as a Bazel BUILD target."""

  def __init__(
      self,
      name: str,
      bazel_target: str,
      generated_so_file: Path,
      target_file: str,
      is_binary: bool = False,
      copy_include_files: bool = False,
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
    self.copy_include_files = copy_include_files


class BuildBazelExtension(build_ext.build_ext):
  """A command that runs Bazel to build a C/C++ extension."""

  def run(self):
    if self.inplace:
      # This corresponds to pip install --editable, and here there is nothing
      # to do. Assumes the user had previously run `bazel build`, and then
      # the python module will automatically detect that it is installed in
      # place and use the development config to find the binaries, etc.
      return

    for ext in self.extensions:
      self.bazel_build(ext)
    # explicitly call `bazel shutdown` for graceful exit
    self.spawn(["bazel", "shutdown"])

  def copy_extensions_to_source(self):
    """Copy generated extensions into the source tree.

    This is done in the ``bazel_build`` method, so it's not necessary to
    do again in the `build_ext` base class.
    """

  def copy_include_files(self):
    """Copy include files from bazel external directory to the libdir."""
    # Find the bazel directory that contains the source files for external
    # dependencies. We copy over the include files for C++ dependencies needed
    # during JIT compilation. It's a symlink and shutil.copytree doesn't work
    # through symlinks, so we have to resolve() it first.

    # Normally there would be an external/ dir, but for some reason the
    # containerized execution doesn't create that symlink.

    # bazel-heir -> /home/user/.cache/bazel/_bazel_j2kun/<hash>/execroot/_main
    bazel_out = (Path(self.build_temp) / "bazel-out").resolve()
    bazel_heir = bazel_out.parent
    out_root = bazel_out.parent.parent.parent
    external = out_root / "external"
    libdir = Path(self.build_lib) / "heir"
    subdirs = [
        Path("openfhe") / "src" / "binfhe" / "include",
        Path("openfhe") / "src" / "core" / "include",
        Path("openfhe") / "src" / "pke" / "include",
        Path("cereal") / "include",
    ]
    dirs_to_copy = {external / subdir: libdir / subdir for subdir in subdirs}

    # Include yosys techmaps.
    dirs_to_copy[
        bazel_heir / "lib" / "Transforms" / "YosysOptimizer" / "yosys"
    ] = (libdir / "techmaps")

    patterns = (
        "*.v",  # techmap files
        "*.h",
        "*.hpp",
        "LICENSE",
        "README.md",
    )

    for src, dst in dirs_to_copy.items():
      print(f"Copying {src} to {dst}")
      shutil.copytree(
          src=src,
          dst=dst,
          ignore=include_patterns(patterns),
          dirs_exist_ok=True,
      )

  def bazel_build(self, ext: BazelExtension) -> None:  # noqa: C901
    """Runs the bazel build to create the package."""
    temp_path = Path(self.build_temp)

    if ext.copy_include_files:
      self.copy_include_files()

    # We round to the minor version, which makes rules_python
    # look up the latest available patch version internally.
    python_version = "{}.{}".format(*sys.version_info[:2])

    cc_choice = (
        [
            # manylinux host that cibuildwheel uses only has gcc by default
            "--action_env=CC=gcc",
            "--action_env=CXX=g++",
            "--host_action_env=CC=gcc",
            "--host_action_env=CXX=g++",
        ]
        if is_cibuildwheel()
        else [
            "--action_env=CC=clang",
            "--action_env=CXX=clang++",
            "--host_action_env=CC=clang",
            "--host_action_env=CXX=clang++",
        ]
    )

    bazel_argv = [
        "bazel",
        "build",
        ext.bazel_target,
        # Using a strict action env in the docker container prevents the build
        # tools from finding python3, since manylinux installs python3 in a
        # nonstandard directory and doesn't symlink it to /usr/bin or
        # /usr/local/bin
        "--noincompatible_strict_action_env",
        # make output suitable for CI
        "--curses=no",
        "--ui_event_filters=ERROR",
        f"--symlink_prefix={temp_path / 'bazel-'}",
        "--compilation_mode=opt",
        f"--cxxopt={'/std:c++17' if IS_WINDOWS else '-std=c++17'}",
        f"--@rules_python//python/config_settings:python_version={python_version}",
    ] + cc_choice

    if IS_WINDOWS:
      # Link with python*.lib.
      for library_dir in self.library_dirs:
        bazel_argv.append("--linkopt=/LIBPATH:" + library_dir)
    elif IS_MAC:
      # C++17 needs macOS 10.14 at minimum
      bazel_argv.append("--macos_minimum_os=10.15")

    with _maybe_patch_toolchains():
      self.spawn(bazel_argv)

    # copy the Bazel build artifacts into setuptools' libdir,
    # from where the wheel is built.
    print("\n\nCopying Bazel build artifacts to setuptools libdir\n\n")
    srcdir = temp_path / "bazel-bin"
    libdir = Path(self.build_lib) / "heir"

    # map from srcdir-relative paths to libdir-relative paths
    srcdir_path = srcdir / ext.generated_so_file
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

      # Also copy binaries to project root so they can be included in data_files
      root_path = Path(ext.target_file)
      print(f"Copying {srcdir_path} to {root_path} for data_files")
      shutil.copyfile(srcdir_path, root_path)
      os.chmod(
          root_path,
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
    package_data={"heir_py": ["py.typed", "*.pyi"]},
    ext_modules=[
        BazelExtension(
            name="heir_py._heir_opt",
            bazel_target="//tools:heir-opt",
            generated_so_file=Path("tools") / "heir-opt",
            target_file="heir-opt",
            py_limited_api=py_limited_api,
            is_binary=True,
        ),
        BazelExtension(
            name="heir_py._heir_translate",
            bazel_target="//tools:heir-translate",
            generated_so_file=Path("tools") / "heir-translate",
            target_file="heir-translate",
            py_limited_api=py_limited_api,
            is_binary=True,
        ),
        BazelExtension(
            name="heir_py._abc",
            bazel_target="@edu_berkeley_abc//:abc",
            generated_so_file=Path("external") / "edu_berkeley_abc" / "abc",
            target_file="abc",
            py_limited_api=py_limited_api,
            is_binary=True,
        ),
        BazelExtension(
            name="heir_py._libopenfhe",
            bazel_target="@openfhe//:libopenfhe",
            generated_so_file=Path("external") / "openfhe" / "libopenfhe.so",
            target_file="libopenfhe.so",
            py_limited_api=py_limited_api,
            copy_include_files=True,
        ),
    ],
    data_files=[("bin", ["heir-opt", "heir-translate"])],
    options=options,
)
