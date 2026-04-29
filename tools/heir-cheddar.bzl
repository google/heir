"""Macros for building CHEDDAR-based FHE targets from HEIR-generated code."""

load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

def cheddar_lib(
        name,
        mlir_src,
        heir_translate_flags = ["--emit-cheddar"],
        extra_deps = [],
        **kwargs):
    """Generate a cc_library from HEIR MLIR targeting the CHEDDAR API.

    Args:
        name: Name of the generated cc_library target.
        mlir_src: The input .mlir file (already lowered to cheddar dialect).
        heir_translate_flags: Flags to pass to heir-translate.
        extra_deps: Additional cc_library deps.
        **kwargs: Additional args forwarded to cc_library.
    """

    # Generate the .cpp file
    cpp_name = name + "_cpp"
    heir_translate(
        name = cpp_name,
        src = mlir_src,
        pass_flags = heir_translate_flags,
        generated_filename = name + ".cpp",
    )

    # Generate the .h file
    header_name = name + "_header"
    heir_translate(
        name = header_name,
        src = mlir_src,
        pass_flags = ["--emit-cheddar-header"],
        generated_filename = name + ".h",
    )

    cc_library(
        name = name,
        srcs = [":" + cpp_name],
        hdrs = [":" + header_name],
        deps = [
            "@cheddar//:cheddar",
            "@cuda//:cuda_headers",
            "@cuda//:cuda_runtime",
            "@cuda//:thrust",
        ] + extra_deps,
        target_compatible_with = requires_cheddar(),
        **kwargs
    )
