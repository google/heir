"""A macro providing an end-to-end test for Cheddar codegen."""

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@heir//tools:heir-cheddar.bzl", _cheddar_lib = "cheddar_lib")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

cheddar_lib = _cheddar_lib

def cheddar_end_to_end_test(
        name,
        mlir_src,
        test_src = None,
        entry_function = None,
        mlir_to_ckks_flags = [],
        scheme_to_cheddar_flags = [],
        heir_opt_flags = [],
        parameter_file = None,
        externalize_constants = False,
        data = [],
        deps = [],
        tags = [],
        **kwargs):
    """A rule for generating Cheddar code and running an end-to-end test on it.

    Args:
      name: Target base name.
      mlir_src: The source mlir file.
      test_src: Optional C++ test harness source file (.cpp / .cc). When
        provided, creates a runnable cc_test with GPU tags. When None, creates a
        build_test compile/link smoke test on the generated library.
      entry_function: Optional entry function name.
      mlir_to_ckks_flags: Optional sub-options for --mlir-to-ckks.
      scheme_to_cheddar_flags: Optional sub-options for --scheme-to-cheddar.
      heir_opt_flags: Optional explicit flag override list for heir-opt.
      parameter_file: Optional parameter file label.
      externalize_constants: Forwarded to heir_opt; when True, appends
        :<name>_heir_opt_ext_consts to cc_test.data.
      data: Data dependencies to pass to cc_test/heir_opt.
      deps: Dependencies to pass to cc_library and cc_test.
      tags: Tags to pass to generated targets.
      **kwargs: Keyword arguments to pass to cc_library and cc_test / build_test.
    """
    cc_lib_target_name = "%s_lib" % name

    TEST_ATTRS = ["args", "env", "env_inherit", "flaky", "local", "shard_count", "size", "timeout"]
    lib_kwargs = {k: v for k, v in kwargs.items() if k not in TEST_ATTRS}

    cheddar_lib(
        name = name,
        mlir_src = mlir_src,
        cc_lib_target_name = cc_lib_target_name,
        entry_function = entry_function,
        mlir_to_ckks_flags = mlir_to_ckks_flags,
        scheme_to_cheddar_flags = scheme_to_cheddar_flags,
        heir_opt_flags = heir_opt_flags,
        parameter_file = parameter_file,
        externalize_constants = externalize_constants,
        data = data,
        deps = deps,
        tags = tags,
        **lib_kwargs
    )

    if test_src:
        gpu_tags = ["requires-gpu-sm90", "exclusive"]
        test_tags = list(gpu_tags)
        for t in tags:
            if t not in test_tags:
                test_tags.append(t)

        cc_test(
            name = name,
            srcs = [test_src],
            deps = deps + [
                ":" + cc_lib_target_name,
                "@cheddar//:cheddar",
                "@googletest//:gtest_main",
            ],
            tags = test_tags,
            data = data,
            **kwargs
        )
    else:
        build_test_kwargs = {k: v for k, v in kwargs.items() if k in TEST_ATTRS}
        build_test(
            name = name,
            targets = [":" + cc_lib_target_name],
            tags = tags,
            **build_test_kwargs
        )
