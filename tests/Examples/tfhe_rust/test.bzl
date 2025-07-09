"""A macro providing an end-to-end test for tfhe-rs codegen."""

load("@heir//tools:heir-tfhe-rs.bzl", "tfhe_rs_lib")
load("@rules_rust//rust:defs.bzl", "rust_test")

def tfhe_rs_end_to_end_test(name, mlir_src, test_src, heir_opt_flags = [], heir_translate_flags = [], data = [], tags = [], deps = [], size = "small", **kwargs):
    """A rule for running generating tfhe-rs and running a test on it.

    Args:
      name: The name of the rust_test target and the generated .rs file basename.
      mlir_src: The source mlir file to run through heir-translate
      test_src: The rust test harness source file.
      heir_opt_flags: Flags to pass to heir-opt before heir-translate
      heir_translate_flags: Flags to pass to heir-translate
      data: Data dependencies to be passed to rust_test/heir_opt
      tags: Tags to pass to rust_test
      deps: Deps to pass to rust_test and cc_library
      size: Size override to pass to rust_test
      **kwargs: Keyword arguments to pass to cc_library and rust_test.
    """
    rs_lib_target_name = "%s_rs_lib" % name
    tfhe_rs_lib(name, mlir_src, rs_lib_target_name, heir_opt_flags, heir_translate_flags, data, tags, deps, **kwargs)
    rust_test(
        name = name,
        srcs = [test_src],
        deps = deps + [
            ":" + rs_lib_target_name,
            "@crates//:tfhe",
        ],
        tags = tags,
        data = data,
        size = size,
        **kwargs
    )
