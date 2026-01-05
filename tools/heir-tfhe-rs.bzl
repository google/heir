"""A macro providing an end-to-end library for tfhe-rs codegen."""

load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_rust//rust:defs.bzl", "rust_library")

def tfhe_rs_lib(name, mlir_src, rs_lib_target_name, heir_opt_flags = [], heir_translate_flags = [], data = [], tags = [], deps = [], **kwargs):
    """A rule for running generating tfhe-rs and running a test on it.

    Args:
      name: The name of the test target and the generated .rs file basename.
      mlir_src: The source mlir file to run through heir-translate
      rs_lib_target_name: The name of the generated rust_library target
      heir_opt_flags: Flags to pass to heir-opt before heir-translate
      heir_translate_flags: Flags to pass to heir-translate
      data: Data dependencies to be passed to heir_opt
      tags: Tags to pass to rust_library
      deps: Deps to pass to rust_library
      **kwargs: Keyword arguments to pass to rust_library.
    """
    rs_codegen_target = name + ".heir_translate_rs"

    generated_rs_filename = "%s_lib.rs" % name
    heir_opt_name = "%s_heir_opt" % name
    generated_heir_opt_name = "%s_heir_opt.mlir" % name

    if heir_opt_flags:
        heir_opt(
            name = heir_opt_name,
            src = mlir_src,
            pass_flags = heir_opt_flags,
            generated_filename = generated_heir_opt_name,
            tags = tags,
            HEIR_YOSYS = True,
            data = ["@heir//lib/Transforms/YosysOptimizer/yosys:techmap_lut3.v", "@heir//lib/Transforms/YosysOptimizer/yosys:techmap_lut4.v"] + data,
        )
    else:
        generated_heir_opt_name = mlir_src

    heir_translate(
        name = rs_codegen_target,
        src = generated_heir_opt_name,
        pass_flags = heir_translate_flags,
        generated_filename = generated_rs_filename,
        tags = tags,
    )
    rust_library(
        name = rs_lib_target_name,
        srcs = [":" + generated_rs_filename],
        deps = deps + [
            "@crates//:serde",
            "@crates//:tfhe",
            "@crates//:rayon",
        ],
        tags = tags,
        data = data,
        **kwargs
    )
