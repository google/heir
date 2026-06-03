"""Configuration of OpenFHE backend."""

import dataclasses
import importlib.resources
import json
import os
from heir.backends.util.common import get_repo_root, is_pip_installed

dataclass = dataclasses.dataclass


def _runfiles_path(toolchain_path: str, path_base: str) -> str:
  """Map a cc-toolchain-reported path to its location in the runfiles tree.

  The cc_toolchain_runtime_info rule reports two flavors of path:
    - `built_in_include_directories` are execroot-relative:
        external/<canonical_repo>/...  (external repo)
        <pkg>/...                      (main repo)
    - `compiler_executable` is a File.short_path:
        ../<canonical_repo>/...        (external repo)
        <pkg>/...                      (main repo)
  In the test runfiles tree both external forms live at
  `<RUNFILES_DIR>/<canonical_repo>/...` and main-repo paths at
  `<RUNFILES_DIR>/_main/<pkg>/...`. Absolute paths are returned unchanged.
  """
  if os.path.isabs(toolchain_path):
    return toolchain_path
  for prefix in ("external" + os.sep, ".." + os.sep):
    if toolchain_path.startswith(prefix):
      return os.path.join(path_base, toolchain_path[len(prefix) :])
  return os.path.join(path_base, "_main", toolchain_path)


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
  - OPENFHE_CXX_TOOLCHAIN_INFO: optional (runfiles-relative) path to the JSON
      file emitted by the //frontend:cc_toolchain_runtime_info rule. It records
      the resolved cc toolchain's `compiler_executable` and
      `built_in_include_directories` (execroot-relative paths), and -- because
      the rule puts the whole toolchain into the test's declared runfiles --
      lets the sandboxed JIT discover and exec the hermetic clang without
      globbing the output_base. The compiler resolved here is a fallback when
      OPENFHE_CXX_COMPILER is not set; its include dirs are appended to
      OPENFHE_CXX_INCLUDE_DIRS.
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
  # The cc_toolchain_runtime_info JSON (runfiles-relative path). It records the
  # resolved hermetic toolchain's compiler + built-in include dirs and, via the
  # rule's runfiles, makes the whole toolchain reachable under the sandbox.
  toolchain_info_path = os.environ.get("OPENFHE_CXX_TOOLCHAIN_INFO", "") or None

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

    # Read the resolved cc toolchain from the cc_toolchain_runtime_info JSON.
    # Its paths are execroot-relative (external/<repo>/... or <pkg>/...) and
    # must be mapped into the runfiles tree. The compiler is a fallback for an
    # unset OPENFHE_CXX_COMPILER; the built-in include dirs (clang resource
    # dir, hermetic glibc / kernel / compiler-rt headers) are added as extra
    # `-isystem` dirs so the JIT replicates the toolchain's own header set.
    if toolchain_info_path:
      with open(os.path.join(path_base, toolchain_info_path)) as f:
        toolchain_info = json.load(f)
      if not cxx_compiler:
        cxx_compiler = _runfiles_path(
            toolchain_info["compiler_executable"], path_base
        )
        if debug:
          print(
              "HEIRpy Debug (OpenFHE Backend): using cc toolchain compiler"
              f" {cxx_compiler}"
          )
      cxx_include_dirs += [
          _runfiles_path(d, path_base)
          for d in toolchain_info["built_in_include_directories"]
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

  # If we were handed hermetic libc++ `-isystem` dirs (and thus the
  # accompanying `-stdlib=libc++ --sysroot=/dev/null -nostdlibinc` flags) but
  # resolved no usable compiler, this is not a recoverable configuration:
  # falling back to the host compiler would feed it those hermetic flags and
  # fail with an opaque "'std' is not a class" / undefined-symbol error far
  # from the real cause, so fail early and precisely here instead. The compiler
  # should normally come from the OPENFHE_CXX_TOOLCHAIN_INFO JSON (a declared
  # runfiles input) or from OPENFHE_CXX_COMPILER.
  if cxx_include_dirs and not cxx_compiler:
    raise RuntimeError(
        "OpenFHE backend was configured with hermetic libc++ include dirs but"
        " resolved no matching toolchain compiler. The host compiler cannot"
        " parse these libc++ headers, so there is no usable fallback; provide"
        " the toolchain via OPENFHE_CXX_TOOLCHAIN_INFO (the"
        " //frontend:cc_toolchain_runtime_info JSON) or set"
        " OPENFHE_CXX_COMPILER explicitly to fix this."
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

  ABI contract: the libopenfhe.so bundled in the pip wheel is built with the
  *system* C++ ABI (libstdc++; setup.py pins the wheel build to the host cc
  toolchain via --extra_toolchains), NOT the hermetic LLVM toolchain's static
  libc++ used for bazel tests. The runtime JIT therefore deliberately uses the
  user's host compiler ($CXX/$CC or clang++/g++ on PATH) with no hermetic
  flags here: the host compiler's default libstdc++ matches the bundled
  library. The cxx_compiler / extra_compiler_flags fields are only populated
  on the bazel-test path (from_os_env), where the prebuilt libopenfhe is
  hermetic-libc++ and the JIT must match it.
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
