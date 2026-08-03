"""A rule for running heir-opt."""

def executable_attr(label):
    """A helper for declaring executable dependencies."""
    return attr.label(
        default = Label(label),
        executable = True,
        # commenting this out breaks cross-compilation, but this should not be a problem
        # for developer builds
        # cfg = "exec",
        cfg = "target",
    )

_HEIR_OPT = "@heir//tools:heir-opt"

def _heir_opt_impl(ctx):
    generated_file = ctx.outputs.generated_filename
    args = ctx.actions.args()
    pass_flags_location_expanded = [ctx.expand_location(flag, ctx.attr.data) for flag in ctx.attr.pass_flags]
    args.add_all(pass_flags_location_expanded)
    args.add_all(["-o", generated_file.path])
    args.add(ctx.file.src)

    outputs = [generated_file]
    res_dir = None
    if ctx.attr.externalize_constants:
        output_dir = ctx.attr.ext_const_output_dir
        if not output_dir:
            output_dir = ctx.label.name + "_resources"
        res_dir = ctx.actions.declare_directory(output_dir)
        outputs.append(res_dir)
        args.add("--ext-const-output-dir=" + res_dir.path)
        runtime_path = res_dir.short_path
        args.add("--ext-const-runtime-load-dir=" + runtime_path)

    env_vars = {}
    if ctx.attr.HEIR_YOSYS:
        # https://bazel.build/remote/output-directories#layout-diagram
        HEIR_BASE_PATH = "_main/"
        runtime_dir = ctx.executable._heir_opt_binary.path + ".runfiles"
        yosys_scripts_dir = runtime_dir + "/" + HEIR_BASE_PATH + "lib/Transforms/YosysOptimizer/yosys"
        abc_path = runtime_dir + "/abc+/abc_bin"
        env_vars["HEIR_YOSYS_SCRIPTS_DIR"] = yosys_scripts_dir
        env_vars["HEIR_ABC_BINARY"] = abc_path

    ctx.actions.run(
        inputs = ctx.attr.src.files,
        mnemonic = "HeirOpt",
        tools = ctx.files.data,
        outputs = outputs,
        arguments = [args],
        env = env_vars,
        executable = ctx.executable._heir_opt_binary,
    )
    runfiles = ctx.runfiles(files = [res_dir] if res_dir else [])
    runfiles = runfiles.merge(ctx.runfiles(collect_default = True))
    return [
        DefaultInfo(
            files = depset([generated_file, ctx.file.src]),
            runfiles = runfiles,
        ),
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
        "pass_flags": attr.string_list(
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
        "externalize_constants": attr.bool(
            doc = """
            Whether to externalize constants.
            """,
            default = True,
        ),
        "ext_const_output_dir": attr.string(
            doc = """
            If set, externalize constants to this directory. Defaults to <target_name>_resources.
            """,
            default = "",
        ),
        "_heir_opt_binary": executable_attr(_HEIR_OPT),
    },
)
