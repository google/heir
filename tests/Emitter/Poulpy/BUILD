load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_rust//rust:defs.bzl", "rust_library")
load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = ["@heir//tests:test_utilities"],
    driver = "@heir//tests:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Compiles the Rust emitted for emit_poulpy.mlir against the pinned poulpy
# crates, so a poulpy API change breaks the build instead of the generated code.
heir_translate(
    name = "emit_poulpy_rs",
    src = "emit_poulpy.mlir",
    generated_filename = "emit_poulpy.rs",
    pass_flags = ["--emit-poulpy"],
)

rust_library(
    name = "emit_poulpy_compiles",
    srcs = ["emit_poulpy.rs"],
    deps = [
        "@crates//:anyhow",
        "@crates//:poulpy-ckks",
        "@crates//:poulpy-core",
        "@crates//:poulpy-cpu-ref",
        "@crates//:poulpy-hal",
    ],
)
