"""A macro providing an end-to-end test for OpenFHE codegen."""

load("@heir//bazel/openfhe:copts.bzl", "OPENMP_COPTS", "OPENMP_LINKOPTS")
load("@heir//tools:heir-openfhe.bzl", "openfhe_lib")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

def openfhe_end_to_end_test(name, mlir_src, test_src, generated_lib_header, heir_opt_flags = [], heir_translate_flags = [], externalize_constants = True, ext_const_output_dir = "", data = [], size = "small", tags = [], deps = [], generate_debug_helper = False, **kwargs):
    """A rule for running generating OpenFHE and running a test on it.

    Args:
      name: The name of the cc_test target and the generated .cc file basename.
      mlir_src: The source mlir file to run through heir-translate
      test_src: The C++ test harness source file.
      generated_lib_header: The name of the generated .h file (explicit
        because it needs to be manually #include'd in the test_src file)
      heir_opt_flags: Flags to pass to heir-opt before heir-translate
      heir_translate_flags: Flags to pass to heir-translate
      externalize_constants: If True, externalize constants.
      ext_const_output_dir: If set, externalize constants to this directory.
      data: Data dependencies to be passed to cc_test/heir_opt
      size: Size to pass to cc_test
      tags: Tags to pass to cc_test
      deps: Deps to pass to cc_test and cc_library
      generate_debug_helper: Flag for generating default debug helper code,
      **kwargs: Keyword arguments to pass to cc_library and cc_test.
    """
    cc_lib_target_name = "%s_cc_lib" % name
    openfhe_lib(name = name, mlir_src = mlir_src, generated_lib_header = generated_lib_header, cc_lib_target_name = cc_lib_target_name, heir_opt_flags = heir_opt_flags, heir_translate_flags = heir_translate_flags, externalize_constants = externalize_constants, ext_const_output_dir = ext_const_output_dir, data = data, tags = tags, deps = deps, generate_debug_helper = generate_debug_helper, **kwargs)
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
        size = size,
        copts = OPENMP_COPTS,
        linkopts = OPENMP_LINKOPTS,
        **kwargs
    )
