"""A macro providing an end-to-end test for OpenFHE codegen."""

load("@heir//bazel/openfhe:copts.bzl", "OPENFHE_LINKOPTS", "OPENMP_COPTS")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")

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
      data: Data dependencies to be passed to cc_test
      tags: Tags to pass to cc_test
      deps: Deps to pass to cc_test and cc_library
      **kwargs: Keyword arguments to pass to cc_library and cc_test.
    """
    cc_codegen_target = name + ".heir_translate_cc"
    h_codegen_target = name + ".heir_translate_h"
    cc_lib_target_name = "%s_cc_lib" % name
    generated_cc_filename = "%s_lib.inc.cc" % name
    heir_opt_name = "%s_heir_opt" % name
    generated_heir_opt_name = "%s_heir_opt.mlir" % name
    heir_translate_flags = heir_translate_flags + ["--emit-openfhe-pke", "--openfhe-include-type=source-relative"]

    if heir_opt_flags:
        heir_opt(
            name = heir_opt_name,
            src = mlir_src,
            pass_flags = heir_opt_flags,
            generated_filename = generated_heir_opt_name,
        )
    else:
        generated_heir_opt_name = mlir_src

    heir_translate(
        name = cc_codegen_target,
        src = generated_heir_opt_name,
        pass_flags = heir_translate_flags,
        generated_filename = generated_cc_filename,
    )
    heir_translate(
        name = h_codegen_target,
        src = generated_heir_opt_name,
        pass_flags = heir_translate_flags,
        generated_filename = generated_lib_header,
    )
    native.cc_library(
        name = cc_lib_target_name,
        srcs = [":" + generated_cc_filename],
        hdrs = [":" + generated_lib_header],
        deps = deps + ["@openfhe//:pke"],
        tags = tags,
        copts = OPENMP_COPTS,
        linkopts = OPENFHE_LINKOPTS,
        **kwargs
    )
    native.cc_test(
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
        copts = OPENMP_COPTS,
        linkopts = OPENFHE_LINKOPTS,
        **kwargs
    )
