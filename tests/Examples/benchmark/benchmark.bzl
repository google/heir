"""Macros providing compiled MLIR for benchmarking"""

load("@heir//tools:heir-opt.bzl", "heir_opt")

def executable_attr(label):
    """A helper for declaring executable dependencies."""
    return attr.label(
        default = Label(label),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    )

_LLC = "@llvm-project//llvm:llc"
_MLIR_TRANSLATE = "@llvm-project//mlir:mlir-translate"

def _binary_impl(ctx):
    generated_file = ctx.outputs.generated_filename
    args = ctx.actions.args()
    args.add_all(ctx.attr.pass_flags)
    args.add_all(["-o", generated_file.path])
    args.add(ctx.file.src)

    ctx.actions.run(
        inputs = ctx.attr.src.files,
        outputs = [generated_file],
        arguments = [args],
        executable = ctx.executable._binary,
        toolchain = None,
    )
    return [
        DefaultInfo(files = depset([generated_file, ctx.file.src])),
    ]

llc = rule(
    doc = """
      This rule runs llc
      """,
    implementation = _binary_impl,
    attrs = {
        "src": attr.label(
            doc = "A single LLVM IR source file to translate.",
            allow_single_file = [".ll"],
        ),
        "pass_flags": attr.string_list(
            doc = """
            The pass flags passed to llc, e.g., --filetype=obj
            """,
        ),
        "generated_filename": attr.output(
            doc = """
            The name used for the output file, including the extension (e.g.,
            <filename>.rs for rust files).
            """,
            mandatory = True,
        ),
        "_binary": executable_attr(_LLC),
    },
)

mlir_translate = rule(
    doc = """
      This rule takes MLIR input and runs mlir-translate on it to produce
      a single generated source file in some target language.
      """,
    implementation = _binary_impl,
    attrs = {
        "src": attr.label(
            doc = "A single MLIR source file to translate.",
            allow_single_file = [".mlir"],
        ),
        "pass_flags": attr.string_list(
            doc = """
            The pass flag passed to mlir-translate, e.g., --mlir-to-llvmir
            """,
        ),
        "generated_filename": attr.output(
            doc = """
            The name used for the output file, including the extension (e.g.,
            <filename>.rs for rust files).
            """,
            mandatory = True,
        ),
        "_binary": executable_attr(_MLIR_TRANSLATE),
    },
)

def heir_benchmark_test(name, mlir_src, test_src, heir_opt_flags = "", data = [], tags = [], deps = [], **kwargs):
    """A rule for running compiling MLIR code and running a test linked to it.

    Args:
      name: The name of the cc_test target.
      mlir_src: The source mlir file to compile.
      test_src: The C++ test harness source file.
      heir_opt_flags: Flags to pass to heir-opt before mlir-translate.
      data: Data dependencies to be passed to cc_test
      tags: Tags to pass to cc_test
      deps: Deps to pass to cc_test and cc_library
      **kwargs: Keyword arguments to pass to cc_library and cc_test.
    """
    heir_opt_name = "%s_heir_opt" % name
    generated_heir_opt_name = "%s_heir_opt.mlir" % name
    llvmir_target = "%s_mlir_translate" % name
    generated_llvmir_name = "%s_llvmir.ll" % name
    obj_name = "%s_object" % name
    generated_obj_name = "%s.o" % name
    import_name = "%s_object_import" % name

    if heir_opt_flags:
        heir_opt(
            name = heir_opt_name,
            src = mlir_src,
            pass_flag = heir_opt_flags,
            generated_filename = generated_heir_opt_name,
        )
    else:
        generated_heir_opt_name = mlir_src

    mlir_translate(
        name = llvmir_target,
        src = generated_heir_opt_name,
        pass_flags = ["--mlir-to-llvmir"],
        generated_filename = generated_llvmir_name,
    )

    llc(
        name = obj_name,
        src = generated_llvmir_name,
        pass_flags = ["-relocation-model=pic", "-filetype=obj"],
        generated_filename = generated_obj_name,
    )

    native.cc_import(
        name = import_name,
        objects = [generated_obj_name],
    )

    native.cc_test(
        name = name,
        srcs = test_src,
        deps = deps + [":" + import_name],
        tags = tags,
        data = data + [generated_obj_name],
        **kwargs
    )
