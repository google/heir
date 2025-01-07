"""A macro providing an end-to-end test for Lattigo codegen."""

load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

def lattigo_end_to_end_test(name, mlir_src, test_src, heir_opt_flags = [], heir_translate_flags = [], data = [], tags = [], deps = [], **kwargs):
    """A rule for running generating Lattigo and running a test on it.

    Args:
      name: The name of the go_test target and the generated .go file basename.
      mlir_src: The source mlir file to run through heir-translate
      test_src: The Go test harness source file.
      heir_opt_flags: Flags to pass to heir-opt before heir-translate
      heir_translate_flags: Flags to pass to heir-translate
      data: Data dependencies to be passed to go_test
      tags: Tags to pass to go_test
      deps: Deps to pass to go_test and go_library
      **kwargs: Keyword arguments to pass to go_library and go_test.
    """
    go_codegen_target = name + ".heir_translate_go"
    go_lib_target_name = "%s_go_lib" % name
    generated_go_filename = "%s_lib.inc.go" % name
    heir_opt_name = "%s_heir_opt" % name
    generated_heir_opt_name = "%s_heir_opt.mlir" % name
    heir_translate_flags = heir_translate_flags + ["--emit-lattigo"]

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
        name = go_codegen_target,
        src = generated_heir_opt_name,
        pass_flags = heir_translate_flags,
        generated_filename = generated_go_filename,
    )
    go_library(
        name = go_lib_target_name,
        srcs = [":" + generated_go_filename],
        deps = deps + [
            "@lattigo//:lattigo",
            "@lattigo//core/rlwe",
            "@lattigo//schemes/bgv",
        ],
        tags = tags,
        **kwargs
    )
    go_test(
        name = name,
        srcs = [test_src, generated_go_filename],
        deps = deps + [
            ":" + go_lib_target_name,
            "@lattigo//:lattigo",
            "@lattigo//core/rlwe",
            "@lattigo//schemes/bgv",
        ],
        tags = tags,
        data = data,
        **kwargs
    )
