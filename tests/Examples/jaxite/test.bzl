"""A macro providing an end-to-end test for jaxite codegen."""

load("@heir//tools:heir-jaxite.bzl", "fhe_jaxite_lib")
load("@rules_python//python:py_test.bzl", "py_test")

def jaxite_end_to_end_test(name, mlir_src, test_src, heir_opt_pass_flags = [], tags = [], deps = [], **kwargs):
    py_lib_target_name = "%s_py_lib" % name
    fhe_jaxite_lib(name, mlir_src, py_lib_target_name = py_lib_target_name, tags = tags, deps = deps, heir_opt_pass_flags = heir_opt_pass_flags, **kwargs)
    py_test(
        name = name,
        srcs = [test_src],
        strict_deps = False,
        main = test_src,
        deps = deps + [
            ":" + py_lib_target_name,
            "@heir_pip_deps//jaxite",
            "@abseil-py//absl/testing:absltest",
        ],
        tags = tags,
    )
