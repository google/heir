"""A macro providing an end-to-end test for OpenFHE codegen."""

load("@heir//tools:heir-translate.bzl", "heir_translate")

def openfhe_end_to_end_test(name, mlir_src, test_src, generated_lib_header, data = [], tags = [], deps = [], **kwargs):
    """A rule for running generating OpenFHE and running a test on it.

    Args:
      name: The name of the cc_test target and the generated .cc file basename.
      mlir_src: The source mlir file to run through heir-translate
      test_src: The C++ test harness source file.
      generated_lib_header: The name of the generated .h file (explicit
        because it needs to be manually #include'd in the test_src file)
      data: Data dependencies to be passed to cc_test
      tags: Tags to pass to cc_test
      deps: Deps to pass to cc_test and cc_library
      **kwargs: Keyword arguments to pass to cc_library and cc_test.
    """
    cc_codegen_target = name + ".heir_translate_cc"
    h_codegen_target = name + ".heir_translate_h"
    cc_lib_target_name = "%s_cc_lib" % name
    generated_cc_filename = "%s_lib.inc.cc" % name

    heir_translate(
        name = cc_codegen_target,
        src = mlir_src,
        pass_flag = "--emit-openfhe-pke",
        generated_filename = generated_cc_filename,
    )
    heir_translate(
        name = h_codegen_target,
        src = mlir_src,
        pass_flag = "--emit-openfhe-pke-header",
        generated_filename = generated_lib_header,
    )
    native.cc_library(
        name = cc_lib_target_name,
        srcs = [":" + generated_cc_filename],
        hdrs = [":" + generated_lib_header],
        deps = deps + ["@openfhe//:pke"],
        tags = tags,
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
        **kwargs
    )
