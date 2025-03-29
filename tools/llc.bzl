"""A rule for running llc."""

def executable_attr(label):
    """A helper for declaring executable dependencies."""
    return attr.label(
        default = Label(label),
        allow_single_file = True,
        executable = True,
        # commenting this out breaks cross-compilation, but this should not be a problem
        # for developer builds
        # cfg = "exec",
        cfg = "target",
    )

_llc = "@llvm-project//llvm:llc"

def _llc_impl(ctx):
    generated_file = ctx.outputs.generated_filename
    args = ctx.actions.args()
    args.add_all(ctx.attr.pass_flags)
    args.add_all(["-o", generated_file.path])
    args.add(ctx.file.src)

    ctx.actions.run(
        inputs = ctx.attr.src.files,
        outputs = [generated_file],
        arguments = [args],
        executable = ctx.executable._llc_binary,
        toolchain = None,
    )
    return [
        DefaultInfo(files = depset([generated_file, ctx.file.src])),
    ]

llc = rule(
    doc = """
      This rule takes LLVMIR input and runs llc on it to produce
      an object file.
      """,
    implementation = _llc_impl,
    attrs = {
        "src": attr.label(
            doc = "A single LLVMIR source file to translate.",
            allow_single_file = [".ll"],
        ),
        "pass_flags": attr.string_list(
            doc = """
            The pass flags passed to llc, e.g., -filetype=obj.
            """,
        ),
        "generated_filename": attr.output(
            doc = """
            The name used for the output file, including the extension (e.g.,
            <filename>.o for object files).
            """,
            mandatory = True,
        ),
        "_llc_binary": executable_attr(_llc),
    },
)
