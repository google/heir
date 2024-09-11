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
    env_vars = {}
    if ctx.attr.HEIR_YOSYS:
        HEIR_BASE_PATH = "heir/"
        runtime_dir = ctx.executable._heir_opt_binary.path + ".runfiles"
        yosys_scripts_dir = runtime_dir + "/" + HEIR_BASE_PATH + "lib/Transforms/YosysOptimizer/yosys"
        abc_path = runtime_dir + "/edu_berkeley_abc/abc"
        env_vars["HEIR_YOSYS_SCRIPTS_DIR"] = yosys_scripts_dir
        env_vars["HEIR_ABC_BINARY"] = abc_path

    ctx.actions.run(
        inputs = ctx.attr.src.files,
        tools = ctx.files.data,
        outputs = [generated_file],
        arguments = [args],
        env = env_vars,
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
        "data": attr.label_list(
            doc = "Additional files needed for running heir-opt. Example: yosys techmap files.",
            allow_files = True,
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
        "HEIR_YOSYS": attr.bool(
            doc = """
            The flag sets the environment variables needed for Yosys and ABC when True.
            """,
            default = False,
        ),
        "_heir_opt_binary": executable_attr(_HEIR_OPT),
    },
)
