"""Helpers for assembling an MLIR-style Python package from bazel targets.

The upstream MLIR bazel overlay only exposes the python sources of the MLIR
bindings as filegroups (rooted at `mlir/python/mlir/...` in the llvm-project
repo) and the C++ extensions as loose cc_binary/cc_library targets. It does not
assemble an importable package, so downstream projects must relocate all the
pieces into a single directory tree themselves.
"""

def _package_relative_path(f):
    """Returns the path of `f` relative to the package that owns it."""
    path = f.short_path
    if path.startswith("../"):
        # External repo: ../<repo_name>/<package>/...
        path = path.split("/", 2)[2]
    pkg = f.owner.package
    if pkg and path.startswith(pkg + "/"):
        path = path[len(pkg) + 1:]
    return path

def _mlir_python_package_impl(ctx):
    prefix = ctx.attr.package_name
    outs = []

    def _emit(f, rel):
        out = ctx.actions.declare_file("%s/%s/%s" % (ctx.label.name, prefix, rel))

        # Copy (rather than symlink) so that `$ORIGIN` rpaths on the shared
        # libraries resolve to siblings in this tree.
        ctx.actions.run_shell(
            inputs = [f],
            outputs = [out],
            command = "cp -f \"$1\" \"$2\"",
            arguments = [f.path, out.path],
            mnemonic = "MlirPyPackageCopy",
            progress_message = "Assembling %s" % out.short_path,
        )
        outs.append(out)

    # Upstream sources live under `mlir/...` relative to their package
    # (llvm-project//mlir/python); rebase them under `prefix/`.
    for f in ctx.files.upstream_srcs:
        rel = _package_relative_path(f)
        if not rel.startswith("mlir/"):
            fail("Unexpected upstream python file layout: %s" % f.short_path)
        _emit(f, rel[len("mlir/"):])

    # Sources already laid out relative to `prefix/` in their own package.
    for f in ctx.files.srcs:
        rel = _package_relative_path(f)
        if rel.startswith(prefix + "/"):
            rel = rel[len(prefix) + 1:]
        _emit(f, rel)

    # Native extensions and shared libraries go into `_mlir_libs/`.
    for f in ctx.files.native_libs:
        _emit(f, "_mlir_libs/" + f.basename)

    return [DefaultInfo(
        files = depset(outs),
        runfiles = ctx.runfiles(files = outs),
    )]

mlir_python_package = rule(
    implementation = _mlir_python_package_impl,
    doc = """Assembles a relocated MLIR python package `<name>/<package_name>/`.

    Use as the `data` of a py_library with `imports = ["<name>"]`.
    """,
    attrs = {
        "package_name": attr.string(mandatory = True),
        "upstream_srcs": attr.label_list(allow_files = True),
        "srcs": attr.label_list(allow_files = True),
        "native_libs": attr.label_list(allow_files = True),
    },
)
