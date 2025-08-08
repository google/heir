"""A macro providing an end-to-end library for OpenFHE BinFHE codegen."""

load("@heir//bazel/openfhe:copts.bzl", "MAYBE_OPENFHE_LINKOPTS", "MAYBE_OPENMP_COPTS")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

def openfhe_binfhe_lib(name, mlir_src, generated_lib_header, cc_lib_target_name, heir_opt_flags = [], heir_translate_flags = [], data = [], tags = [], deps = [], **kwargs):
    """A rule for generating OpenFHE BinFHE library from MLIR.

    Args:
      name: The name of the cc_test target and the generated .cc file basename.
      mlir_src: The source mlir file to run through heir-translate
      generated_lib_header: The name of the generated .h file (explicit
        because it needs to be manually #include'd in the test_src file)
      cc_lib_target_name: The name of the generated cc_library target
      heir_opt_flags: Flags to pass to heir-opt before heir-translate
      heir_translate_flags: Flags to pass to heir-translate
      data: Data dependencies to be passed to heir_opt
      tags: Tags to pass to cc_test and cc_library
      deps: Deps to pass to cc_test and cc_library
      **kwargs: Keyword arguments to pass to cc_library and cc_test.
    """
    cc_codegen_target = name + ".heir_translate_cc"
    h_codegen_target = name + ".heir_translate_h"

    generated_cc_filename = "%s_lib.inc.cc" % name
    heir_opt_name = "%s_heir_opt" % name
    generated_heir_opt_name = "%s_heir_opt.mlir" % name

    # Use BinFHE-specific translation flags
    heir_translate_flags = heir_translate_flags + ["--emit-openfhe-bin"]

    if heir_opt_flags:
        heir_opt(
            name = heir_opt_name,
            src = mlir_src,
            pass_flags = heir_opt_flags,
            generated_filename = generated_heir_opt_name,
            data = data,
        )
    else:
        generated_heir_opt_name = mlir_src

    heir_translate(
        name = cc_codegen_target,
        src = generated_heir_opt_name,
        pass_flags = heir_translate_flags,
        generated_filename = generated_cc_filename,
    )

    # For BinFHE header generation, only use the header flag
    heir_translate(
        name = h_codegen_target,
        src = generated_heir_opt_name,
        pass_flags = ["--emit-openfhe-bin-header"],
        generated_filename = generated_lib_header,
    )
    cc_library(
        name = cc_lib_target_name,
        srcs = [":" + generated_cc_filename],
        hdrs = [":" + generated_lib_header],
        deps = deps + ["@openfhe//:binfhe"],
        tags = tags,
        data = data,
        copts = MAYBE_OPENMP_COPTS,
        linkopts = MAYBE_OPENFHE_LINKOPTS,
        **kwargs
    )
