"""A macro providing an end-to-end test for Lattigo codegen."""

load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_go//go:def.bzl", "go_library")

def heir_lattigo_lib(name, mlir_src, go_library_name = None, heir_opt_flags = [], heir_translate_flags = [], extra_srcs = [], data = [], tags = [], deps = [], **kwargs):
    """A rule for generating Lattigo code from an MLIR file.

    Args:
      name: The name of the generated go_library target and package name
      mlir_src: The source mlir file to run through heir-translate
      go_library_name: The name of the generated go library and package
      heir_opt_flags: Flags to pass to heir-opt before heir-translate
      heir_translate_flags: Flags to pass to heir-translate
      extra_srcs: Extra sources to pass to go_library
      data: Data dependencies to be passed to go_library
      tags: Tags to pass to go_library
      deps: Deps to pass to  and go_library
      **kwargs: Keyword arguments to pass to go_library
    """
    go_codegen_target = name + ".heir_translate_go"
    go_package_name = go_library_name or name
    generated_go_filename = "%s_lib.go" % go_package_name
    heir_opt_name = "%s_heir_opt" % name
    generated_heir_opt_name = "%s_heir_opt.mlir" % name
    heir_translate_flags = heir_translate_flags + ["--emit-lattigo", "--package-name=" + go_package_name]

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
        name = go_codegen_target,
        src = generated_heir_opt_name,
        pass_flags = heir_translate_flags,
        generated_filename = generated_go_filename,
    )
    go_library(
        name = go_package_name,
        srcs = extra_srcs + [":" + generated_go_filename],
        deps = deps + [
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
        ],
        tags = tags,
        data = data,
        **kwargs
    )
