"""A rule for running mlir-opt with the GPU to NVVM pipeline."""

def executable_attr(label):
    """A helper for declaring executable dependencies."""
    return attr.label(
        default = Label(label),
        allow_single_file = True,
        executable = True,
        cfg = "target",
    )

_MLIR_OPT = "@llvm-project//mlir:mlir-opt"

def _mlir_opt_gpu_impl(ctx):
    generated_file = ctx.outputs.generated_filename
    args = ctx.actions.args()
    pipeline_flag = "-gpu-lower-to-nvvm-pipeline=cubin-chip=%s cubin-format=%s" % (
        ctx.attr.chip,
        ctx.attr.format,
    )
    args.add(pipeline_flag)
    args.add_all(ctx.attr.pass_flags)
    args.add_all(["-o", generated_file.path])
    args.add(ctx.file.src)

    ctx.actions.run(
        mnemonic = "MLIROptGPU",
        inputs = ctx.attr.src.files,
        outputs = [generated_file],
        tools = ctx.files.data,
        arguments = [args],
        executable = ctx.executable._mlir_opt_binary,
        toolchain = None,
    )
    return [
        DefaultInfo(files = depset([generated_file, ctx.file.src])),
    ]

mlir_opt_gpu = rule(
    doc = """
      This rule runs mlir-opt with -gpu-lower-to-nvvm-pipeline to produce
      an MLIR module containing embedded GPU binary objects and host LLVM IR.
      """,
    implementation = _mlir_opt_gpu_impl,
    attrs = {
        "src": attr.label(
            doc = "A single MLIR source file to optimize.",
            allow_single_file = [".mlir"],
            mandatory = True,
        ),
        "pass_flags": attr.string_list(
            doc = "Additional pass flags to pass to mlir-opt.",
            default = [],
        ),
        "chip": attr.string(
            doc = "The GPU chip architecture, e.g. sm_90.",
            default = "sm_90",
        ),
        "format": attr.string(
            doc = "The cubin compilation format, e.g. isa or fatbin.",
            default = "isa",
        ),
        "data": attr.label_list(
            doc = "Additional data/tool dependencies for mlir-opt.",
            allow_files = True,
            default = [],
        ),
        "generated_filename": attr.output(
            doc = "The name used for the output file (e.g. <filename>.mlir).",
            mandatory = True,
        ),
        "_mlir_opt_binary": executable_attr(_MLIR_OPT),
    },
)
