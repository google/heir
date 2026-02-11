"""A rule for running heir-translate."""

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

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

_HEIR_TRANSLATE = "@heir//tools:heir-translate"

def _heir_translate_impl(ctx):
    generated_file = ctx.outputs.generated_filename
    args = ctx.actions.args()
    args.add_all(ctx.attr.pass_flags)
    args.add_all(["-o", generated_file.path])
    args.add(ctx.file.src)

    ctx.actions.run(
        inputs = ctx.attr.src.files,
        outputs = [generated_file],
        mnemonic = "HeirTranslate",
        arguments = [args],
        executable = ctx.executable._heir_translate_binary,
        toolchain = None,
    )

    cc_info = CcInfo(
        compilation_context = cc_common.create_compilation_context(
            includes = depset([generated_file.dirname]),
        ),
    )

    return [
        DefaultInfo(files = depset([generated_file, ctx.file.src])),
        cc_info,
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
        "pass_flags": attr.string_list(
            doc = """
            The pass flags passed to heir-translate, e.g., --emit-openfhe-pke
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
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
)
