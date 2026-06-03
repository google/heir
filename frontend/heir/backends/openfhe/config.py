"""Configuration of OpenFHE backend."""

import dataclasses
import glob
import importlib.resources
import os
from heir.backends.util.common import get_repo_root, is_pip_installed

dataclass = dataclasses.dataclass


def _discover_hermetic_toolchain_clang(libcxx_include_dir: str) -> str | None:
  """Find the hermetic LLVM toolchain's clang++ next to its libc++ headers.

  When libopenfhe.so was built by the hermetic LLVM toolchain (libc++), the
  JIT step must use that *same* toolchain's clang to compile against the
  matching libc++ headers -- the host clang/g++ is generally too old to parse
  the toolchain's libc++ (e.g. "libc++ only supports Clang 20 and later").

  We are handed the libc++ `-isystem` directory (a bazel `copy_to_directory`
  output under <output_base>/.../bin/external/llvm++llvm_source+libcxx/...,
  reached through the test's runfiles). Resolve it and walk up to the bazel
  `output_base`, then glob for the sibling toolchain repo's clang++:
      <output_base>/external/llvm++_repo_rules+llvm-toolchain-*/bin/clang++

  Returns the clang++ path if found, else None (caller falls back to host).
  """
  real = os.path.realpath(libcxx_include_dir)
  # .../<output_base>/execroot/<ws>/bazel-out/<cfg>/bin/external/<repo>/<dir>
  marker = os.sep + "execroot" + os.sep
  idx = real.find(marker)
  if idx == -1:
    return None
  output_base = real[:idx]
  matches = sorted(
      glob.glob(
          os.path.join(
              output_base,
              "external",
              "llvm++_repo_rules+llvm-toolchain-*",
              "bin",
              "clang++",
          )
      )
  )
  # Pick the lexicographically *last* match: if a toolchain version bump has
  # left a stale `llvm-toolchain-<old>` repo dir behind in the output_base, the
  # higher version sorts last, so we prefer the newer clang over the stale one.
  return matches[-1] if matches else None


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
    cxx_compiler: Optional path to the C++ compiler the JIT step should use. If
      None, the JIT falls back to $CXX/$CC or whatever clang++/g++ is on PATH.
      This matters when the prebuilt libopenfhe.so was compiled against a
      specific C++ standard library (e.g. a hermetic toolchain's libc++): the
      JIT-compiled .so must use the *same* standard library so that the
      mangled OpenFHE symbols (std::__1::... for libc++ vs std::... for
      libstdc++) resolve at load time.
    extra_compiler_flags: Extra flags appended to the JIT compiler invocation
      (e.g. ["-stdlib=libc++", "-nostdinc++", "-isystem", "<libc++ headers>"]).
      Used to match the standard library / ABI of the prebuilt OpenFHE library.
    extra_linker_search_paths: Extra -L search paths for the JIT link step
      (e.g. the directory containing the toolchain's libc++.so / libc++abi.so).
    extra_link_libs: Extra -l libraries for the JIT link step (e.g. ["c++",
      "c++abi"]) to satisfy the standard-library runtime.
  """

  include_dirs: list[str]
  include_type: str
  lib_dir: str
  link_libs: list[str]
  cxx_compiler: str | None = None
  extra_compiler_flags: list[str] = dataclasses.field(default_factory=list)
  extra_linker_search_paths: list[str] = dataclasses.field(default_factory=list)
  extra_link_libs: list[str] = dataclasses.field(default_factory=list)


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
  - OPENFHE_CXX_COMPILER: optional (runfiles-relative) path to the C++ compiler
      the JIT step should use. Needed when libopenfhe.so was built with a
      specific toolchain whose C++ standard library must be matched (e.g. a
      hermetic clang + libc++).
  - OPENFHE_CXX_FLAGS: colon-separated raw extra compiler flags (e.g.
      "-stdlib=libc++:-nostdinc++"). NOT runfiles-resolved.
  - OPENFHE_CXX_INCLUDE_DIRS: colon-separated extra (runfiles-resolved) include
      dirs emitted as `-isystem` (e.g. the toolchain's libc++ headers).
  - OPENFHE_CXX_LIB_DIRS: colon-separated extra (runfiles-resolved) `-L` linker
      search paths (e.g. the dir holding libc++.so / libc++abi.so).
  - OPENFHE_CXX_LINK_LIBS: colon-separated extra `-l` libraries (e.g. "c++:c++abi")
      to satisfy the standard-library runtime.
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

  # JIT-compiler / standard-library matching (see OpenFHEConfig docstring).
  cxx_compiler = os.environ.get("OPENFHE_CXX_COMPILER", "") or None
  extra_compiler_flags = os.environ.get("OPENFHE_CXX_FLAGS", "").split(":")
  cxx_include_dirs = os.environ.get("OPENFHE_CXX_INCLUDE_DIRS", "").split(":")
  extra_linker_search_paths = os.environ.get("OPENFHE_CXX_LIB_DIRS", "").split(
      ":"
  )
  extra_link_libs = os.environ.get("OPENFHE_CXX_LINK_LIBS", "").split(":")

  # remove empty strings from lists
  include_dirs = [dir for dir in include_dirs if dir]
  link_libs = [lib for lib in link_libs if lib]
  extra_compiler_flags = [f for f in extra_compiler_flags if f]
  cxx_include_dirs = [dir for dir in cxx_include_dirs if dir]
  extra_linker_search_paths = [d for d in extra_linker_search_paths if d]
  extra_link_libs = [lib for lib in extra_link_libs if lib]

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
    # The toolchain's compiler / libc++ headers / libc++ libs (added as data
    # deps) are likewise relative to the runfiles root.
    if cxx_compiler:
      cxx_compiler = os.path.join(path_base, cxx_compiler)
    cxx_include_dirs = [os.path.join(path_base, d) for d in cxx_include_dirs]
    extra_linker_search_paths = [
        os.path.join(path_base, d) for d in extra_linker_search_paths
    ]

  for include_dir in include_dirs:
    if not os.path.exists(include_dir):
      print(
          f'Warning: OpenFHE include directory "{include_dir}" does not exist'
      )

  # The libc++ / libc++abi `-isystem` dirs are `copy_to_directory` tree
  # artifacts handed in via $(rootpath ...) + RUNFILES_DIR. If one is missing or
  # empty (e.g. the tree did not land in the test runfiles) the JIT compile
  # fails with a confusing "'initializer_list' file not found", so check up
  # front and surface the real problem.
  for cxx_include_dir in cxx_include_dirs:
    if not os.path.isdir(cxx_include_dir):
      print(
          "Warning: C++ stdlib include directory"
          f' "{cxx_include_dir}" does not exist (libc++ headers will not be'
          " found by the JIT compiler)"
      )

  # If we were handed hermetic libc++ headers but no explicit compiler, find
  # the toolchain's own clang next to them (the host compiler is generally too
  # old to parse the toolchain's libc++ headers).
  if cxx_include_dirs and not cxx_compiler:
    cxx_compiler = _discover_hermetic_toolchain_clang(cxx_include_dirs[0])
    if cxx_compiler:
      if debug:
        print(
            "HEIRpy Debug (OpenFHE Backend): discovered hermetic toolchain"
            f" clang at {cxx_compiler}"
        )
    else:
      # We were configured with hermetic libc++ `-isystem` dirs (and thus the
      # accompanying `-stdlib=libc++ --sysroot=/dev/null -nostdlibinc` flags)
      # but could not locate the matching toolchain clang. Falling back to the
      # host compiler would feed it those hermetic flags and fail with an
      # opaque "'std' is not a class" / undefined-symbol error far from the
      # real cause, so surface the actual problem here.
      print(
          "Warning: OpenFHE backend was configured with hermetic libc++"
          " include dirs but could not discover the matching toolchain clang"
          f" (searched the output_base derived from {cxx_include_dirs[0]!r})."
          " The JIT will fall back to the host compiler, which cannot parse"
          " these libc++ headers; set OPENFHE_CXX_COMPILER explicitly to fix"
          " this."
      )

  # Extra libc++ (or other stdlib) include dirs are emitted as `-isystem`
  # before any default search paths so they take precedence over a host stdlib.
  for cxx_include_dir in cxx_include_dirs:
    extra_compiler_flags += ["-isystem", cxx_include_dir]

  # If something has been found from the environment variables, return it
  if include_dirs:
    return OpenFHEConfig(
        include_dirs=include_dirs,
        lib_dir=lib_dir,
        link_libs=link_libs,
        include_type=include_type,
        cxx_compiler=cxx_compiler,
        extra_compiler_flags=extra_compiler_flags,
        extra_linker_search_paths=extra_linker_search_paths,
        extra_link_libs=extra_link_libs,
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
          str(package_path / "openfhe+"),
          str(package_path / "openfhe+" / "src" / "binfhe" / "include"),
          str(package_path / "openfhe+" / "src" / "core" / "include"),
          str(package_path / "openfhe+" / "src" / "pke" / "include"),
          str(package_path / "cereal+" / "include"),
      ],
      include_type="source-relative",
      lib_dir=str(package_path),
      link_libs=["openfhe"],
  )
