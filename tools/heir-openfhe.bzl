"""A macro providing an end-to-end library for OpenFHE codegen."""

load("@heir//bazel/openfhe:copts.bzl", "MAYBE_OPENFHE_LINKOPTS", "MAYBE_OPENMP_COPTS")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

def openfhe_lib(
        name,
        mlir_src,
        generated_lib_header,
        cc_lib_target_name = None,
        pybind_target_name = None,
        heir_opt_flags = [],
        heir_translate_flags = [],
        data = [],
        tags = [],
        deps = [],
        **kwargs):
    """A rule for running generating OpenFHE and running a test on it.

    Args:
      name: The name of the cc_test target and the generated .cc file basename.
      mlir_src: The source mlir file to run through heir-translate
      generated_lib_header: The name of the generated .h file (explicit
        because it needs to be manually #include'd in the test_src file)
      cc_lib_target_name: The name of the generated cc_library target
      pybind_target_name: The name of the generated pybind_extension target
      heir_opt_flags: Flags to pass to heir-opt before heir-translate
      heir_translate_flags: Flags to pass to heir-translate
      data: Data dependencies to be passed to heir_opt
      tags: Tags to pass to cc_test and cc_library
      deps: Deps to pass to cc_test and cc_library
      **kwargs: Keyword arguments to pass to cc_library and cc_test.
    """
    cc_codegen_target = name + ".heir_translate_cc"
    h_codegen_target = name + ".heir_translate_h"
    pybind_codegen_target = name + ".heir_translate_pybind"
    generated_cc_filename = "%s_lib.inc.cc" % name
    heir_opt_name = "%s_heir_opt" % name
    generated_heir_opt_name = "%s_heir_opt.mlir" % name
    heir_translate_cc_flags = heir_translate_flags + ["--emit-openfhe-pke", "--openfhe-include-type=source-relative"]
    heir_translate_h_flags = heir_translate_flags + ["--emit-openfhe-pke-header", "--openfhe-include-type=source-relative"]

    cc_lib_target = cc_lib_target_name
    if not cc_lib_target:
        cc_lib_target = "_heir_%s" % name

    pybind_target = pybind_target_name
    if not pybind_target:
        pybind_target = "_heir_%s" % name

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
        pass_flags = heir_translate_cc_flags,
        generated_filename = generated_cc_filename,
    )
    heir_translate(
        name = h_codegen_target,
        src = generated_heir_opt_name,
        pass_flags = heir_translate_h_flags,
        generated_filename = generated_lib_header,
    )

    cc_library(
        name = cc_lib_target,
        srcs = [generated_cc_filename],
        hdrs = [generated_lib_header],
        deps = deps + ["@openfhe//:pke"],
        tags = tags,
        data = data,
        copts = MAYBE_OPENMP_COPTS,
        linkopts = MAYBE_OPENFHE_LINKOPTS,
        **kwargs
    )

    # add Python bindings on top
    generated_pybind_cc_name = "%s_bindings.cpp" % name
    pybind_flags = heir_translate_flags + [
        "--openfhe-include-type=source-relative",
        "--emit-openfhe-pke-pybind",
        "--pybind-header-include=%s" % generated_lib_header,
        # The module name here needs to match the target name,
        # as this is what pybind_extension uses for the name
        # exposed to `import`
        "--pybind-module-name=%s" % pybind_target,
    ]
    heir_translate(
        name = pybind_codegen_target,
        src = generated_heir_opt_name,
        pass_flags = pybind_flags,
        generated_filename = generated_pybind_cc_name,
    )
    pybind_extension(
        name = pybind_target,
        srcs = [generated_pybind_cc_name],
        deps = [
            pybind_codegen_target,
            "@openfhe//:pke",
            cc_lib_target,
        ],
        tags = tags,
        data = data,
        **kwargs
    )
