"""A rule for running heir-opt."""

def executable_attr(label):
    """A helper for declaring executable dependencies."""
    return attr.label(
        default = Label(label),
        executable = True,
        cfg = "exec",
    )

_HEIR_OPT = "@heir//tools:heir-opt"

def _heir_opt_impl(ctx):
    generated_file = ctx.outputs.generated_filename
    args = ctx.actions.args()
    args.add(ctx.attr.pass_flag)
    args.add_all(["-o", generated_file.path])
    args.add(ctx.file.src)
    ctx.actions.run(
        inputs = ctx.attr.src.files,
        outputs = [generated_file],
        arguments = [args],
        executable = ctx.executable._heir_opt_binary,
    )
    return [
        DefaultInfo(files = depset([generated_file, ctx.file.src])),
    ]

heir_opt = rule(
    doc = """
      This rule takes MLIR input and runs heir-opt on it to produce
      a single output file after applying the given MLIR passes.
      """,
    implementation = _heir_opt_impl,
    attrs = {
        "src": attr.label(
            doc = "A single MLIR source file to opt.",
            allow_single_file = [".mlir"],
        ),
        "pass_flag": attr.string(
            doc = """
            The pass flags passed to heir-opt, e.g., --canonicalize
            """,
        ),
        "generated_filename": attr.output(
            doc = """
            The name used for the output file, including the extension (e.g.,
            <filename>.mlir).
            """,
            mandatory = True,
        ),
        "_heir_opt_binary": executable_attr(_HEIR_OPT),
    },
)
