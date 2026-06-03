"""A rule that exports the resolved C++ toolchain to the runtime JIT.

The frontend's OpenFHE backend JIT-compiles emitted C++ at test runtime. When
libopenfhe.so was built with the hermetic LLVM toolchain (libc++), that JIT step
must use the *same* toolchain's clang against the matching libc++ headers.

Rather than discovering the toolchain by globbing the bazel output_base (which
breaks under sandboxing because the toolchain repo is not a declared input),
this rule resolves the configured cc toolchain via Starlark, writes its compiler
path and built-in include directories to a JSON file (the data dep), and -- via
`cc.all_files` in runfiles -- brings the whole toolchain (clang binary, wrapper
scripts, resource dir, sysroot headers) into the test's declared runfiles so the
JIT can exec it under the sandbox.
"""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")

def _cc_toolchain_runtime_info_impl(ctx):
    cc = find_cpp_toolchain(ctx)

    # cc.compiler_executable is the cc_toolchain's compiler *wrapper*
    # (e.g. external/llvm+/toolchain/gcc). That wrapper is a build-time-only
    # artifact: it is NOT among cc.all_files and so never lands in runfiles, so
    # the JIT cannot exec it under the sandbox. The actual driver binary
    # (bin/clang++) IS in cc.all_files, so report its runfiles-relative
    # (execroot-relative) path instead. We hand the JIT the toolchain's hermetic
    # flags / include dirs explicitly (see frontend/testing.bzl), so the raw
    # driver -- without the wrapper's flag injection -- is exactly what we want.
    compiler = cc.compiler_executable
    for f in cc.all_files.to_list():
        if f.basename in ("clang++", "clang") and f.dirname.endswith("/bin"):
            compiler = f.short_path
            if f.basename == "clang++":
                break

    info = {
        "compiler_executable": compiler,
        "built_in_include_directories": cc.built_in_include_directories,
    }
    out = ctx.actions.declare_file(ctx.label.name + ".json")
    ctx.actions.write(out, json.encode(info))
    return [DefaultInfo(
        files = depset([out]),
        runfiles = ctx.runfiles(files = [out], transitive_files = cc.all_files),
    )]

cc_toolchain_runtime_info = rule(
    implementation = _cc_toolchain_runtime_info_impl,
    toolchains = use_cpp_toolchain(),
    fragments = ["cpp"],
)
