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

    # TODO(#729): Remove after upstream dialect loader is fixed
    tmp = ctx.actions.declare_file("dummy_prepended_" + ctx.file.src.basename)
    args.add(tmp)
    ctx.actions.run_shell(
        inputs = [ctx.file.src],
        outputs = [tmp],
        command = "printf '%s\n%s\n' \"#dummy = #polynomial.ring<coefficientType = i32>\" \"$(cat \"$1\")\" > \"$2\"",
        arguments = [ctx.file.src.path, tmp.path],
    )

    ctx.actions.run(
        inputs = [tmp],
        outputs = [generated_file],
        arguments = [args],
        executable = ctx.executable._heir_translate_binary,
        toolchain = None,
    )
    return [
        DefaultInfo(files = depset([generated_file, tmp, ctx.file.src])),
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
