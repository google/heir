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

    # libc++/libc++abi alone are not enough: a raw clang has none of the rest of
    # the toolchain's include/sysroot config that the bazel cc_toolchain wrapper
    # injects, so parsing the libc++ headers fails with "'std' is not a class,
    # namespace, or enumeration". Replicate the toolchain's full hermetic header
    # set by also handing the JIT (a) the hermetic glibc headers, (b) the Linux
    # UAPI kernel headers, and (c) the clang builtins resource dir. These are the
    # same `-isystem` roots `bazel aquery 'mnemonic("CppCompile", @openfhe//:core)'`
    # shows the toolchain using. We reference version-agnostic apex aliases so a
    # toolchain bump does not require editing version-pathed repo names:
    #   @glibc//:glibc_headers_directory          -> .../glibc_headers_*/include
    #   @kernel_headers//:kernel_headers_directory -> .../linux_kernel_headers_*/include
    #   @llvm//:builtin_resource_dir               -> .../lib/clang/<N>  (headers in /include)
    glibc_isystem_dir = "@glibc//:glibc_headers_directory"
    kernel_isystem_dir = "@kernel_headers//:kernel_headers_directory"
    clang_resource_dir = "@llvm//:builtin_resource_dir"

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
            glibc_isystem_dir,
            kernel_isystem_dir,
            clang_resource_dir,
        ],
        shard_count = 3,
        # The runtime JIT must invoke the *hermetic* clang (only it can parse
        # this toolchain's libc++ headers and its builtin paths match the
        # -isystem resource dir above; the host clang/g++ silently mis-resolves
        # and fails with "'std' is not a class"). config.py locates that clang by
        # globbing the bazel output_base's `external/llvm++...-toolchain-*/bin`.
        # That toolchain repo is not declared as a runfiles input, so under the
        # default test sandbox it is invisible and the JIT falls back to the host
        # clang. Run these tests unsandboxed so the JIT can reach (and exec) the
        # hermetic clang on the real output_base. (Bringing the clang binary into
        # runfiles is not an option: the toolchain repo's clang filegroups are
        # private to that repo and not re-exported by @llvm.)
        tags = tags + ["no-sandbox"],
        env = {
            # this dir is relative to $RUNFILES_DIR, which is set by bazel at runtime
            "OPENFHE_LIB_DIR": "openfhe+",
            "OPENFHE_INCLUDE_TYPE": "source-relative",
            "OPENFHE_LINK_LIBS": ":".join(libs),
            "OPENFHE_INCLUDE_DIR": ":".join(include_dirs),
            # Match libopenfhe.so's libc++ ABI in the runtime JIT step, and (like
            # the toolchain's own compiles) ignore the host sysroot / default
            # stdlib include paths so ONLY the hermetic -isystem roots below are
            # used. `-nostdlibinc` is the toolchain's own flag (it drops the C++
            # and C standard-library include dirs but keeps the clang builtins);
            # `--sysroot=/dev/null` removes the host sysroot.
            "OPENFHE_CXX_FLAGS": ":".join([
                "-stdlib=libc++",
                "-nostdinc++",
                "--sysroot=/dev/null",
                "-nostdlibinc",
            ]),
            # Use $(rlocationpath ...), NOT $(rootpath ...): for a cross-repo
            # target $(rootpath) yields a `../<repo>/...` path that, joined onto
            # RUNFILES_DIR in config.py, escapes the runfiles tree
            # (runfiles/../llvm++...) so the -isystem dir doesn't exist and even
            # <initializer_list> is "not found". $(rlocationpath) gives the
            # runfiles-relative `<repo>/...` path that resolves correctly.
            # The clang builtins live in the resource dir's `include`
            # subdirectory; `@llvm//:builtin_resource_dir` points at the resource
            # dir itself (.../lib/clang/<N>), so append /include.
            "OPENFHE_CXX_INCLUDE_DIRS": ":".join([
                "$(rlocationpath %s)" % libcxx_isystem_dir,
                "$(rlocationpath %s)" % libcxxabi_isystem_dir,
                "$(rlocationpath %s)" % glibc_isystem_dir,
                "$(rlocationpath %s)" % kernel_isystem_dir,
                "$(rlocationpath %s)/include" % clang_resource_dir,
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
