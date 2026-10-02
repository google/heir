"""A macro providing an end-to-end compile/link test for CHEDDAR codegen."""

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

def cheddar_end_to_end_test(
        name,
        mlir_src,
        heir_opt_flags = ["--cheddar-to-emitc"],
        heir_translate_flags = ["--mlir-to-cpp"],
        deps = [],
        tags = [],
        **kwargs):
    """Generates C++ from a cheddar-dialect MLIR file and links it against CHEDDAR.

    The generated code is compiled and linked, but never run, so no GPU is
    needed. Code generation (`<name>_codegen`) runs in every configuration; the
    compile/link test (`<name>`) requires `--config=cheddar` (see
    .github/workflows/build_cheddar.yml).

    Args:
      name: The name of the build_test target; prefixes the generated targets.
      mlir_src: The source mlir file to run through heir-opt and heir-translate.
      heir_opt_flags: Flags to pass to heir-opt before heir-translate.
      heir_translate_flags: Flags to pass to heir-translate.
      deps: Deps to pass to the cc_binary.
      tags: Tags to pass to the generated targets.
      **kwargs: Keyword arguments to pass to the cc_binary.
    """
    emitc_mlir = name + "_emitc.mlir"
    generated_cc = name + ".cc"
    main_cc = name + "_main.cc"

    heir_opt(
        name = name + "_heir_opt",
        src = mlir_src,
        pass_flags = heir_opt_flags,
        generated_filename = emitc_mlir,
        externalize_constants = False,
        tags = tags,
    )
    heir_translate(
        name = name + "_heir_translate",
        src = ":" + emitc_mlir,
        pass_flags = heir_translate_flags,
        generated_filename = generated_cc,
        tags = tags,
    )
    build_test(
        name = name + "_codegen",
        targets = [":" + generated_cc],
        tags = tags,
    )

    # The generated code has no main; a stub makes it a linkable binary.
    write_file(
        name = name + "_main",
        out = main_cc,
        content = ["int main() { return 0; }", ""],
        tags = tags,
    )
    cc_binary(
        name = name + "_bin",
        srcs = [
            ":" + generated_cc,
            ":" + main_cc,
        ],
        # Under -c opt, --gc-sections lets the linker drop the (unreferenced)
        # generated functions without resolving their symbols. Keep them live so
        # the link checks every CHEDDAR symbol the generated code uses.
        linkopts = ["-Wl,--no-gc-sections"],
        target_compatible_with = requires_cheddar(),
        deps = deps + ["@cheddar//:cheddar"],
        tags = tags,
        **kwargs
    )
    build_test(
        name = name,
        targets = [":" + name + "_bin"],
        target_compatible_with = requires_cheddar(),
        tags = tags,
    )
