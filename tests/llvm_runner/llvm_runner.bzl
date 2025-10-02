"""A macro providing an end-to-end test for Plaintext Backend."""

load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:llc.bzl", "llc")
load("@heir//tools:mlir-translate.bzl", "mlir_translate")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

def executable_attr():
    """A helper for declaring executable dependencies."""
    return attr.label(
        allow_single_file = True,
        executable = True,
        # commenting this out breaks cross-compilation, but this should not be a problem
        # for developer builds
        # cfg = "exec",
        cfg = "target",
    )

def _binary_impl(ctx):
    generated_file = ctx.outputs.generated_filename
    args = ctx.actions.args()
    args.add(generated_file.path)

    ctx.actions.run(
        mnemonic = "BinaryRule",
        executable = ctx.executable.binary,
        arguments = [args],
        outputs = [generated_file],
        toolchain = None,
    )
    return [
        DefaultInfo(files = depset([generated_file])),
    ]

binary_rule = rule(
    doc = """
      This rule runs the generated binary.
      """,
    implementation = _binary_impl,
    attrs = {
        "generated_filename": attr.output(
            doc = """
            The name used for the output file.
            """,
            mandatory = True,
        ),
        "binary": executable_attr(),
    },
)

def llvm_runner_test(
        name,
        mlir_src,
        heir_opt_flags,
        mlir_translate_flags = None,
        llc_flags = None,
        deps = [],
        defines = [],
        main_c_src = None,
        log_file_name = None,
        log_file_visibility = None,
        data = []):
    """Define a lit test for the Plaintext Backend.

    Args:
      name: The name of the test.
      mlir_src: The source mlir file to run through.
      heir_opt_flags: Flags to pass to heir-opt.
      mlir_translate_flags: Flags to pass to mlir-translate.
      llc_flags: Flags to pass to llc.
      deps: Deps to pass to cc_test.
      defines: Defines to pass to cc_test.
      main_c_src: A C source file containing the main function for the test.
      log_file_name: The name of the log file.
      log_file_visibility: Visibility of the log file.
      data: Data deps to pass to heir_opt
    """
    default_deps = [
        "@heir//tests/llvm_runner:memrefCopy",
    ]

    test_deps = [
        "@googletest//:gtest_main",
    ]
    if log_file_name != None:
        # Can't use gunit main because a debug test needs a custom main
        test_deps = []

    test_deps.extend(default_deps)
    test_deps.extend(deps)

    heir_opt_name = "%s_heir_opt" % name
    generated_heir_opt_name = "%s_heir_opt.mlir" % name

    heir_opt(
        name = heir_opt_name,
        src = mlir_src,
        pass_flags = heir_opt_flags,
        generated_filename = generated_heir_opt_name,
        data = data,
    )

    mlir_translate_name = "%s_mlir_translate" % name
    generated_mlir_translate_name = "%s_mlir_translate.ll" % name

    _mlir_translate_flags = [
        "--mlir-to-llvmir",
    ]
    if mlir_translate_flags != None:
        _mlir_translate_flags = mlir_translate_flags

    mlir_translate(
        name = mlir_translate_name,
        src = generated_heir_opt_name,
        pass_flags = _mlir_translate_flags,
        generated_filename = generated_mlir_translate_name,
    )

    llc_name = "%s_llc" % name
    generated_llc_name = "%s_llc.o" % name

    _llc_flags = [
        "-filetype=obj",
        "-relocation-model=pic",
    ]
    if llc_flags != None:
        _llc_flags = llc_flags

    llc(
        name = llc_name,
        src = generated_mlir_translate_name,
        pass_flags = _llc_flags,
        generated_filename = generated_llc_name,
    )

    cc_test_name = "%s_cc_test" % name
    srcs = [":" + generated_llc_name]
    if main_c_src:
        srcs.append(main_c_src)
    cc_test(
        name = cc_test_name,
        srcs = srcs,
        deps = test_deps,
        defines = defines,
        copts = ["-fPIC"],
    )

    # the following part is for exporting the log for other tests
    if log_file_name != None:
        cc_binary_name = "%s_cc_binary" % name
        cc_binary(
            name = cc_binary_name,
            srcs = srcs,
            deps = test_deps,
            defines = defines,
            copts = ["-fPIC"],
            testonly = True,
        )

        binary_rule_name = "%s_binary_rule" % name

        binary_rule(
            name = binary_rule_name,
            binary = ":" + cc_binary_name,
            generated_filename = log_file_name,
            visibility = log_file_visibility,
            testonly = True,
        )
