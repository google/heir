"""A macro providing an end-to-end test for OpenFHE codegen."""

load("@heir//bazel/openfhe:copts.bzl", "MAYBE_OPENFHE_LINKOPTS", "MAYBE_OPENMP_COPTS")
load("@heir//tools:heir-openfhe.bzl", "openfhe_lib")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

def openfhe_end_to_end_test(name, mlir_src, test_src, generated_lib_header, heir_opt_flags = [], heir_translate_flags = [], data = [], tags = [], deps = [], **kwargs):
    """A rule for running generating OpenFHE and running a test on it.

    Args:
      name: The name of the cc_test target and the generated .cc file basename.
      mlir_src: The source mlir file to run through heir-translate
      test_src: The C++ test harness source file.
      generated_lib_header: The name of the generated .h file (explicit
        because it needs to be manually #include'd in the test_src file)
      heir_opt_flags: Flags to pass to heir-opt before heir-translate
      heir_translate_flags: Flags to pass to heir-translate
      data: Data dependencies to be passed to cc_test/heir_opt
      tags: Tags to pass to cc_test
      deps: Deps to pass to cc_test and cc_library
      **kwargs: Keyword arguments to pass to cc_library and cc_test.
    """
    cc_lib_target_name = "%s_cc_lib" % name
    openfhe_lib(name = name, mlir_src = mlir_src, generated_lib_header = generated_lib_header, cc_lib_target_name = cc_lib_target_name, heir_opt_flags = heir_opt_flags, heir_translate_flags = heir_translate_flags, data = data, tags = tags, deps = deps, **kwargs)
    cc_test(
        name = name,
        srcs = [test_src],
        deps = deps + [
            ":" + cc_lib_target_name,
            "@openfhe//:pke",
            "@openfhe//:core",
            "@googletest//:gtest_main",
        ],
        tags = tags,
        data = data,
        copts = MAYBE_OPENMP_COPTS,
        linkopts = MAYBE_OPENFHE_LINKOPTS,
        **kwargs
    )

def openfhe_interpreter_test(name, mlir_src, test_src, generated_heir_opt_filename = "", heir_opt_flags = [], data = [], tags = [], deps = [], copts = [], timeout = "moderate", **kwargs):
    """A rule for running generating OpenFHE dialect and exposing it to an interpreter.

    Args:
      name: The name of the cc_test target and the generated .cc file basename.
      mlir_src: The source mlir file to run through heir-translate
      test_src: The C++ test harness source file.
      heir_opt_flags: Flags to pass to heir-opt before heir-translate
      generated_heir_opt_filename: The filename of the file output by heir-opt
      data: Data dependencies to be passed to cc_test/heir_opt
      tags: Tags to pass to cc_test
      deps: Deps to pass to cc_test
      copts: Additional copts to pass to cc_test
      timeout: Timeout to pass to cc_test
      **kwargs: Keyword arguments to pass to cc_test.
    """
    heir_opt_name = "%s_heir_opt" % name
    if not generated_heir_opt_filename:
        generated_heir_opt_filename = "%s_heir_opt.mlir" % name

    heir_opt(
        name = heir_opt_name,
        src = mlir_src,
        pass_flags = heir_opt_flags,
        generated_filename = generated_heir_opt_filename,
        data = data,
    )

    cc_test(
        name = name,
        srcs = [test_src],
        deps = deps + [
            "@heir//lib/Target/OpenFhePke:Interpreter",
            "@openfhe//:pke",
            "@openfhe//:core",
            "@googletest//:gtest_main",
            # for mlir source file parsing
            "@llvm-project//mlir:Support",
        ],
        timeout = timeout,
        tags = tags,
        data = data + [":" + generated_heir_opt_filename],
        copts = MAYBE_OPENMP_COPTS + copts,
        linkopts = MAYBE_OPENFHE_LINKOPTS,
        **kwargs
    )
