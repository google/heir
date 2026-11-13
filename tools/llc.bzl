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
_opt = "@llvm-project//llvm:opt"

def sanitizer_opt_flags():
    """Returns `opt` flags that instrument LLVM IR for the active sanitizer.

    Object files produced by `llc` bypass the C++ toolchain, so they are not
    compiled with the sanitizer flags of the current configuration. Under
    MemorySanitizer this causes false positives: uninstrumented code never
    updates shadow memory, so any memory it writes (including heap buffers it
    allocates and memref descriptors it returns) still appears uninitialized
    to instrumented callers.

    The flags below mirror what clang does for `-fsanitize=memory`
    (`-fsanitize-memory-param-retval` is the clang default, hence
    `-msan-eager-checks`). Every function is marked `sanitize_memory` so that
    uses of uninitialized values inside the generated code are reported too.

    Returns:
      A select() evaluating to a (possibly empty) list of `opt` flags.
    """
    return select({
        Label("@heir//tools:msan_build"): [
            # IR emitted by mlir-translate has no target triple, but MSan
            # needs one to pick its shadow memory layout. MSan builds only
            # target x86_64 Linux.
            "-mtriple=x86_64-unknown-linux-gnu",
            "-passes=forceattrs,msan",
            "-force-attribute=sanitize_memory",
            "-msan-eager-checks",
        ],
        "//conditions:default": [],
    }) + select({
        # --config=msan-track-origins uses -fsanitize-memory-track-origins,
        # which defaults to level 2.
        Label("@heir//tools:msan_track_origins_build"): [
            "-msan-track-origins=2",
        ],
        "//conditions:default": [],
    })

def _llc_impl(ctx):
    generated_file = ctx.outputs.generated_filename
    src = ctx.file.src

    if ctx.attr.opt_flags:
        instrumented = ctx.actions.declare_file(ctx.label.name + ".instrumented.ll")
        opt_args = ctx.actions.args()
        opt_args.add_all(ctx.attr.opt_flags)
        opt_args.add_all(["-S", "-o", instrumented.path])
        opt_args.add(src)
        ctx.actions.run(
            mnemonic = "LLVMOptRule",
            inputs = [src],
            outputs = [instrumented],
            arguments = [opt_args],
            executable = ctx.executable._opt_binary,
            toolchain = None,
        )
        src = instrumented

    args = ctx.actions.args()
    args.add_all(ctx.attr.pass_flags)
    args.add_all(["-o", generated_file.path])
    args.add(src)

    ctx.actions.run(
        mnemonic = "LLCRule",
        inputs = [src],
        outputs = [generated_file],
        arguments = [args],
        executable = ctx.executable._llc_binary,
        toolchain = None,
    )
    return [
        DefaultInfo(files = depset([generated_file, ctx.file.src])),
    ]

_llc_rule = rule(
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
        "opt_flags": attr.string_list(
            doc = """
            If non-empty, `opt` is run with these flags on `src` before llc.
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
        "_opt_binary": executable_attr(_opt),
    },
)

def llc(name, sanitize = True, **kwargs):
    """Runs llc on LLVMIR input to produce an object file.

    Args:
      name: The name of the target.
      sanitize: Whether to instrument the IR for the active sanitizer (if any)
        before running llc, so that the resulting object file can be linked
        into sanitizer builds without false positives. See
        `sanitizer_opt_flags`.
      **kwargs: Forwarded to the underlying rule (`src`, `pass_flags`,
        `generated_filename`, ...).
    """
    _llc_rule(
        name = name,
        opt_flags = sanitizer_opt_flags() if sanitize else [],
        **kwargs
    )
