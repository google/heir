"""A macro providing an end-to-end test for Lattigo codegen."""

load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_go//go:def.bzl", "go_library")

def _make_split_preprocessing_libs(utils_name, generated_heir_opt_name, heir_translate_flags, common_deps, tags, data):
    utils_go_filename = utils_name + ".go"
    heir_translate(
        name = utils_name + "_translate",
        src = generated_heir_opt_name,
        pass_flags = heir_translate_flags + ["--emit-lattigo-preprocessing", "--package-name=" + utils_name],
        generated_filename = utils_go_filename,
    )
    go_library(
        name = utils_name,
        srcs = [":" + utils_go_filename],
        deps = common_deps,
        tags = tags,
        data = data,
    )

def heir_lattigo_lib(name, mlir_src, go_library_name = None, heir_opt_flags = [], heir_translate_flags = [], externalize_constants = True, ext_const_output_dir = "", extra_srcs = [], data = [], tags = [], deps = [], split_preprocessing = True, entry_interface = False, **kwargs):
    """A rule for generating Lattigo code from an MLIR file.

    Args:
      name: The name of the generated go_library target and package name
      mlir_src: The source mlir file to run through heir-translate
      go_library_name: The name of the generated go library and package
      heir_opt_flags: Flags to pass to heir-opt before heir-translate
      heir_translate_flags: Flags to pass to heir-translate
      externalize_constants: If True, externalize constants.
      ext_const_output_dir: If set, externalize constants to this directory.
      extra_srcs: Extra sources to pass to go_library
      data: Data dependencies to be passed to go_library
      tags: Tags to pass to go_library
      deps: Deps to pass to  and go_library
      split_preprocessing: Whether to split preprocessing into a separate library
      entry_interface: Whether to also emit the Go facade over the generated
        ABI. Requires a module carrying the entry roles, i.e. one that went
        through --add-client-interface.
      **kwargs: Keyword arguments to pass to go_library
    """
    go_package_name = go_library_name or name
    heir_opt_name = "%s_heir_opt" % name
    generated_heir_opt_name = "%s_heir_opt.mlir" % name

    if heir_opt_flags or externalize_constants:
        heir_opt(
            name = heir_opt_name,
            src = mlir_src,
            pass_flags = heir_opt_flags,
            generated_filename = generated_heir_opt_name,
            data = data,
            externalize_constants = externalize_constants,
            ext_const_output_dir = ext_const_output_dir,
        )
    else:
        generated_heir_opt_name = mlir_src

    common_deps = deps + [
        "@com_github_tuneinsight_lattigo_v6//:lattigo",
        "@com_github_tuneinsight_lattigo_v6//core/rlwe",
        "@com_github_tuneinsight_lattigo_v6//schemes/bgv",
        "@com_github_tuneinsight_lattigo_v6//schemes/ckks",
        "@com_github_tuneinsight_lattigo_v6//circuits/ckks/lintrans",
        "@com_github_tuneinsight_lattigo_v6//circuits/ckks/polynomial",
        "@com_github_tuneinsight_lattigo_v6//utils/bignum",
        "@com_github_tuneinsight_lattigo_v6//utils",
        "@com_github_tuneinsight_lattigo_v6//ring",
        "@com_github_tuneinsight_lattigo_v6//circuits/ckks/bootstrapping",
    ]

    heir_opt_data = data
    if heir_opt_flags or ext_const_output_dir:
        heir_opt_data = data + [":" + heir_opt_name]

    if split_preprocessing == True:
        utils_name = go_package_name + "_utils"
        _make_split_preprocessing_libs(utils_name, generated_heir_opt_name, heir_translate_flags, common_deps, tags, heir_opt_data)
        common_deps = common_deps + [utils_name]
        utils_import_path = native.package_name() + "/" + utils_name
        heir_translate_flags = heir_translate_flags + ["--extra-imports=" + utils_import_path, "--emit-lattigo-preprocessed"]
    else:
        heir_translate_flags = heir_translate_flags + ["--emit-lattigo"]

    generated_go_filename = "%s_lib.go" % go_package_name
    heir_translate_flags = heir_translate_flags + ["--package-name=" + go_package_name]
    heir_translate(
        name = name + ".heir_translate_go",
        src = generated_heir_opt_name,
        pass_flags = heir_translate_flags,
        generated_filename = generated_go_filename,
    )

    generated_srcs = [":" + generated_go_filename]
    if entry_interface:
        # The facade over the generated ABI, in the same package as the library.
        generated_interface_filename = "%s_interface.go" % go_package_name
        heir_translate(
            name = name + ".heir_translate_interface",
            src = generated_heir_opt_name,
            pass_flags = [
                flag
                for flag in heir_translate_flags
                if flag not in ("--emit-lattigo", "--emit-lattigo-preprocessed")
            ] + ["--emit-lattigo-interface"],
            generated_filename = generated_interface_filename,
        )
        generated_srcs.append(":" + generated_interface_filename)

    go_library(
        name = go_package_name,
        srcs = extra_srcs + generated_srcs,
        deps = common_deps,
        tags = tags,
        data = heir_opt_data if not split_preprocessing else data,
        **kwargs
    )
