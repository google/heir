"""A macro providing an end-to-end test for OpenFHE BinFHE codegen."""

load("@heir//bazel/openfhe:copts.bzl", "MAYBE_OPENFHE_LINKOPTS", "MAYBE_OPENMP_COPTS")
load("@heir//tools:heir-openfhe-binfhe.bzl", "openfhe_binfhe_lib")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

def openfhe_binfhe_end_to_end_test(name, mlir_src, test_src, generated_lib_header, heir_opt_flags = [], heir_translate_flags = [], data = [], tags = [], deps = [], **kwargs):
    """A rule for running generating OpenFHE BinFHE and running a test on it.

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
    openfhe_binfhe_lib(name, mlir_src, generated_lib_header, cc_lib_target_name, heir_opt_flags, heir_translate_flags, data, tags, deps, **kwargs)
    cc_test(
        name = name,
        srcs = [test_src],
        deps = deps + [
            ":" + cc_lib_target_name,
            "@openfhe//:binfhe",
            "@openfhe//:core",
            "@googletest//:gtest_main",
        ],
        tags = tags,
        data = data,
        copts = MAYBE_OPENMP_COPTS,
        linkopts = MAYBE_OPENFHE_LINKOPTS,
        **kwargs
    )
