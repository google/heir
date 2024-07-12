"""A macro providing an end-to-end test for jaxite codegen."""

load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

def jaxite_end_to_end_test(name, mlir_src, test_src, tags = [], deps = [], **kwargs):
    """A rule for running generating OpenFHE and running a test on it.

    Args:
      name: The name of the py_test target and the generated .cc file basename.
      mlir_src: The source mlir file to run through heir-translate
      test_src: The C++ test harness source file.
      tags: Tags to pass to py_test
      deps: Deps to pass to py_test and py_library
      **kwargs: Keyword arguments to pass to py_library and py_test.
    """
    py_codegen_target = name + ".heir_translate_py"
    py_lib_target_name = "%s_py_lib" % name
    generated_py_filename = "%s_lib.py" % name

    heir_translate(
        name = py_codegen_target,
        src = mlir_src,
        pass_flag = "--emit-jaxite",
        generated_filename = generated_py_filename,
    )
    py_library(
        name = py_lib_target_name,
        srcs = [":" + generated_py_filename],
        deps = deps + ["@heir_pip_deps_jaxite//:pkg"],
        tags = tags,
        **kwargs
    )
    py_test(
        name = name,
        srcs = [test_src],
        main = test_src,
        deps = deps + [
            ":" + py_lib_target_name,
            "@heir_pip_deps_jaxite//:pkg",
            "@com_google_absl_py//absl/testing:absltest",
        ],
    )
