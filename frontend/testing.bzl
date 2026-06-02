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
    # We hand the JIT compiler the toolchain's own libc++ / libc++abi header
    # search dirs (the same `copy_to_directory` -isystem roots the toolchain uses
    # for its own compiles) plus -stdlib=libc++ -nostdinc++. The libc++ runtime
    # symbols themselves are satisfied at load time by libopenfhe.so, which
    # statically embeds and (weakly) re-exports libc++.
    libcxx_isystem_dir = "@libcxx//:libcxx_headers_include_search_directory"
    libcxxabi_isystem_dir = "@libcxxabi//:libcxxabi_headers_include_search_directory"

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
        data = data + [
            libcxx_isystem_dir,
            libcxxabi_isystem_dir,
        ],
        shard_count = 3,
        tags = tags,
        env = {
            # this dir is relative to $RUNFILES_DIR, which is set by bazel at runtime
            "OPENFHE_LIB_DIR": "openfhe+",
            "OPENFHE_INCLUDE_TYPE": "source-relative",
            "OPENFHE_LINK_LIBS": ":".join(libs),
            "OPENFHE_INCLUDE_DIR": ":".join(include_dirs),
            # Match libopenfhe.so's libc++ ABI in the runtime JIT step.
            "OPENFHE_CXX_FLAGS": ":".join(["-stdlib=libc++", "-nostdinc++"]),
            "OPENFHE_CXX_INCLUDE_DIRS": ":".join([
                "$(rootpath %s)" % libcxx_isystem_dir,
                "$(rootpath %s)" % libcxxabi_isystem_dir,
            ]),
            "HEIR_REPO_ROOT_MARKER": ".",
            "HEIR_OPT_PATH": "tools/heir-opt",
            "HEIR_TRANSLATE_PATH": "tools/heir-translate",
            "PYBIND11_INCLUDE_PATH": "pybind11/include",
            "HEIR_YOSYS_SCRIPTS_DIR": "lib/Transforms/YosysOptimizer/yosys",
            "HEIR_ABC_BINARY": "$(rootpath @abc//:abc_bin)",
            "NUMBA_USE_LEGACY_TYPE_SYSTEM": "1",
        },
    )
