"""A macro providing a library for Cheddar codegen."""

load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

def cheddar_lib(
        name,
        mlir_src,
        cc_lib_target_name = None,
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
    """Generates Cheddar C++ code from MLIR and compiles it to a cc_library.

    Args:
      name: Target base name.
      mlir_src: The source mlir file to run through heir-opt and heir-translate.
      cc_lib_target_name: The name of the generated cc_library target.
      entry_function: Optional entry function name.
      mlir_to_ckks_flags: Optional sub-options for --mlir-to-ckks.
      scheme_to_cheddar_flags: Optional sub-options for --scheme-to-cheddar.
      heir_opt_flags: Optional explicit flag override list for heir-opt.
      parameter_file: Optional parameter file label.
      externalize_constants: Forwarded to heir_opt.
      data: Data dependencies to pass to heir_opt and cc_library.
      deps: Dependencies for the generated cc_library.
      tags: Tags to pass to generated targets.
      **kwargs: Keyword arguments to pass to cc_library.
    """
    heir_opt_name = "%s_heir_opt" % name
    generated_heir_opt_name = "%s_emitc.mlir" % name
    generated_cc_filename = "%s.cc" % name

    if not heir_opt_flags:
        heir_opt_flags = ["--annotate-module=backend=cheddar scheme=ckks"]

        ckks_subopts = list(mlir_to_ckks_flags)
        if parameter_file:
            ckks_subopts.append("parameter-file=$(location %s)" % parameter_file)
        if ckks_subopts:
            heir_opt_flags.append("--mlir-to-ckks=%s" % " ".join(ckks_subopts))
        else:
            heir_opt_flags.append("--mlir-to-ckks")

        cheddar_subopts = list(scheme_to_cheddar_flags)
        if entry_function:
            cheddar_subopts.append("entry-function=%s" % entry_function)
        if cheddar_subopts:
            heir_opt_flags.append("--scheme-to-cheddar=%s" % " ".join(cheddar_subopts))
        else:
            heir_opt_flags.append("--scheme-to-cheddar")

        heir_opt_flags.append("--cheddar-to-emitc")

    heir_opt_data = list(data)
    if parameter_file:
        heir_opt_data.append(parameter_file)

    heir_opt(
        name = heir_opt_name,
        src = mlir_src,
        pass_flags = heir_opt_flags,
        generated_filename = generated_heir_opt_name,
        data = heir_opt_data,
        externalize_constants = externalize_constants,
    )

    heir_translate(
        name = "%s_heir_translate" % name,
        src = ":" + generated_heir_opt_name,
        pass_flags = ["--mlir-to-cpp"],
        generated_filename = generated_cc_filename,
    )

    cc_lib_target = cc_lib_target_name or ("%s_lib" % name)
    cc_lib_data = list(data)
    if externalize_constants:
        cc_lib_data.append(":" + heir_opt_name)

    cc_library(
        name = cc_lib_target,
        srcs = [":" + generated_cc_filename],
        data = cc_lib_data,
        deps = ["@cheddar//:cheddar"] + deps,
        tags = tags,
        **kwargs
    )
