"""A rule for running heir-translate."""

def executable_attr(label):
    """A helper for declaring executable dependencies."""
    return attr.label(
        default = Label(label),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    )

_HEIR_TRANSLATE = "@heir//tools:heir-translate"

def _heir_translate_impl(ctx):
    generated_file = ctx.outputs.generated_filename
    args = ctx.actions.args()
    args.add(ctx.attr.pass_flag)
    args.add_all(["-o", generated_file.path])
    args.add(ctx.file.src)

    ctx.actions.run(
        inputs = ctx.attr.src.files,
        outputs = [generated_file],
        arguments = [args],
        executable = ctx.executable._heir_translate_binary,
        toolchain = None,
    )
    return [
        DefaultInfo(files = depset([generated_file, ctx.file.src])),
    ]

heir_translate = rule(
    doc = """
      This rule takes MLIR input and runs heir-translate on it to produce
      a single generated source file in some target language.
      """,
    implementation = _heir_translate_impl,
    attrs = {
        "src": attr.label(
            doc = "A single MLIR source file to translate.",
            allow_single_file = [".mlir"],
        ),
        "pass_flag": attr.string(
            doc = """
            The pass flag passed to heir-translate, e.g., --emit-openfhe-pke
            """,
        ),
        "generated_filename": attr.output(
            doc = """
            The name used for the output file, including the extension (e.g.,
            <filename>.rs for rust files).
            """,
            mandatory = True,
        ),
        "_heir_translate_binary": executable_attr(_HEIR_TRANSLATE),
    },
)
