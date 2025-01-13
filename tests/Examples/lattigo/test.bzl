"""A macro providing an end-to-end test for Lattigo codegen."""

load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@io_bazel_rules_go//go:def.bzl", "go_library")

def heir_lattigo_lib(name, mlir_src, heir_opt_flags = [], heir_translate_flags = [], data = [], tags = [], deps = [], **kwargs):
    """A rule for generating Lattigo code from an MLIR file.

    Args:
      name: The name of the generated go_library target and package name
      mlir_src: The source mlir file to run through heir-translate
      heir_opt_flags: Flags to pass to heir-opt before heir-translate
      heir_translate_flags: Flags to pass to heir-translate
      data: Data dependencies to be passed to go_library
      tags: Tags to pass to go_library
      deps: Deps to pass to  and go_library
      **kwargs: Keyword arguments to pass to go_library
    """
    go_codegen_target = name + ".heir_translate_go"
    generated_go_filename = "%s_lib.go" % name
    heir_opt_name = "%s_heir_opt" % name
    generated_heir_opt_name = "%s_heir_opt.mlir" % name
    heir_translate_flags = heir_translate_flags + ["--emit-lattigo", "--package-name=" + name]

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
        name = name,
        srcs = [":" + generated_go_filename],
        deps = deps + [
            "@lattigo//:lattigo",
            "@lattigo//core/rlwe",
            "@lattigo//schemes/bgv",
        ],
        tags = tags,
        data = data,
        **kwargs
    )
