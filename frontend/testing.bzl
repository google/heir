"""Macros for py_test rules that use the python frontend."""

load("@rules_python//python:defs.bzl", "py_test")

def frontend_test(name, srcs, deps = [], data = [], tags = []):
    """A py_test replacement with an env including all backend dependencies.

    Args:
      name: the name of the test target.
      srcs: the test source files.
      deps: extra deps to add to the py_test.
      data: extra data deps to add to the py_test.
      tags: tags to add to the py_test.
    """
    include_dirs = [
        "openfhe+/",
        "openfhe+/src/binfhe/include",
        "openfhe+/src/core/include",
        "openfhe+/src/pke/include",
        "cereal+/include",
        "rapidjson+/include",
    ]

    libs = [
        "openfhe",
    ]

    # The prebuilt @openfhe//:libopenfhe is compiled by the hermetic LLVM
    # toolchain against libc++, so its exported symbols use libc++'s
    # std::__1::... mangling. The frontend's runtime JIT step
    # (heir/backends/util/cpp_compiler.py) must therefore also build the emitted
    # C++ against libc++ -- otherwise (e.g. on Linux where the host clang/g++
    # defaults to libstdc++) the OpenFHE symbols are mangled as std::... and fail
    # to resolve, producing an "undefined symbol" ImportError at module load.
    #
    # All of this hermetic-libc++ matching is Linux-only: that is where the
    # prebuilt libopenfhe.so is built with the hermetic toolchain. On macOS (and
    # any other host) the frontend keeps the pre-branch behavior -- the JIT uses
    # the host compiler with no hermetic flags and no Linux-only data deps.
    #
    # On Linux we hand the JIT the resolved cc toolchain via
    # //frontend:cc_toolchain_runtime_info: a small rule that writes the
    # toolchain's compiler path + built-in include dirs to a JSON file (the
    # OPENFHE_CXX_TOOLCHAIN_INFO data dep) and -- crucially -- pulls the whole
    # toolchain (clang binary, wrapper scripts, resource dir, sysroot/glibc/
    # kernel headers) into the test's *declared* runfiles. That makes the
    # hermetic clang reachable under the default test sandbox, so these tests run
    # sandboxed (no more `no-sandbox` tag, no more output_base globbing in
    # config.py).
    #
    # The toolchain's built_in_include_directories already supply the clang
    # builtins resource dir and the hermetic glibc / kernel / compiler-rt header
    # roots, so config.py feeds those to the JIT automatically. libc++ /
    # libc++abi headers are NOT part of the cc toolchain's built-in include dirs,
    # so we still wire them in explicitly via OPENFHE_CXX_INCLUDE_DIRS. We
    # reference version-agnostic apex aliases so a toolchain bump does not require
    # editing version-pathed repo names.
    toolchain_info = "//frontend:cc_toolchain_runtime_info"
    libcxx_isystem_dir = "@libcxx//:libcxx_headers_include_search_directory"
    libcxxabi_isystem_dir = "@libcxxabi//:libcxxabi_headers_include_search_directory"

    linux = "@platforms//os:linux"

    py_test(
        name = name,
        srcs = srcs,
        python_version = "PY3",
        srcs_version = "PY3",
        deps = deps + [
            ":frontend",
            "@abseil-py//absl/testing:absltest",
            "@abc//:abc_bin",
        ],
        imports = ["."],
        data = data + select({
            linux: [
                toolchain_info,
                libcxx_isystem_dir,
                libcxxabi_isystem_dir,
            ],
            "//conditions:default": [],
        }),
        shard_count = 3,
        tags = tags,
        env = {
            # this dir is relative to $RUNFILES_DIR, which is set by bazel at runtime
            "OPENFHE_LIB_DIR": "openfhe+",
            "OPENFHE_INCLUDE_TYPE": "source-relative",
            "OPENFHE_LINK_LIBS": ":".join(libs),
            "OPENFHE_INCLUDE_DIR": ":".join(include_dirs),
            "HEIR_REPO_ROOT_MARKER": ".",
            "HEIR_OPT_PATH": "tools/heir-opt",
            "HEIR_TRANSLATE_PATH": "tools/heir-translate",
            "PYBIND11_INCLUDE_PATH": "pybind11/include",
            "HEIR_YOSYS_SCRIPTS_DIR": "lib/Transforms/YosysOptimizer/yosys",
            "HEIR_ABC_BINARY": "$(rootpath @abc//:abc_bin)",
            "NUMBA_USE_LEGACY_TYPE_SYSTEM": "1",
        } | select({
            linux: {
                # Match libopenfhe.so's libc++ ABI in the runtime JIT step, and
                # (like the toolchain's own compiles) ignore the host sysroot /
                # default stdlib include paths so ONLY the hermetic -isystem roots
                # are used. `-nostdlibinc` is the toolchain's own flag (it drops
                # the C++ and C standard-library include dirs but keeps the clang
                # builtins); `--sysroot=/dev/null` removes the host sysroot.
                "OPENFHE_CXX_FLAGS": ":".join([
                    "-stdlib=libc++",
                    "-nostdinc++",
                    "--sysroot=/dev/null",
                    "-nostdlibinc",
                ]),
                # The cc_toolchain_runtime_info JSON. config.py reads the resolved
                # compiler + built-in include dirs from it and, because the rule's
                # runfiles carry the whole toolchain, the sandboxed JIT can exec
                # the hermetic clang. Use $(rlocationpath ...) so the path joins
                # cleanly onto RUNFILES_DIR.
                "OPENFHE_CXX_TOOLCHAIN_INFO": "$(rlocationpath %s)" % toolchain_info,
                # Use $(rlocationpath ...), NOT $(rootpath ...): for a cross-repo
                # target $(rootpath) yields a `../<repo>/...` path that, joined
                # onto RUNFILES_DIR in config.py, escapes the runfiles tree so the
                # -isystem dir doesn't exist. $(rlocationpath) gives the
                # runfiles-relative `<repo>/...` path that resolves correctly.
                "OPENFHE_CXX_INCLUDE_DIRS": ":".join([
                    "$(rlocationpath %s)" % libcxx_isystem_dir,
                    "$(rlocationpath %s)" % libcxxabi_isystem_dir,
                ]),
            },
            "//conditions:default": {},
        }),
    )
