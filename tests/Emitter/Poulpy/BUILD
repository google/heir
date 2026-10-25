load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_rust//rust:defs.bzl", "rust_library", "rust_test")
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
        "@crates//:poulpy-hal",
    ],
)

# The same check for tensor-form IR after poulpy-bufferize.
heir_opt(
    name = "emit_poulpy_bufferized_opt",
    src = "emit_poulpy_bufferized.mlir",
    generated_filename = "emit_poulpy_bufferized_opt.mlir",
    pass_flags = ["--poulpy-bufferize"],
)

heir_translate(
    name = "emit_poulpy_bufferized_rs",
    src = ":emit_poulpy_bufferized_opt.mlir",
    generated_filename = "emit_poulpy_bufferized.rs",
    pass_flags = ["--emit-poulpy"],
)

rust_library(
    name = "emit_poulpy_bufferized_compiles",
    srcs = ["emit_poulpy_bufferized.rs"],
    deps = [
        "@crates//:anyhow",
        "@crates//:poulpy-ckks",
        "@crates//:poulpy-core",
        "@crates//:poulpy-hal",
    ],
)

# The emitted code is generic over the backend; instantiate it with concrete ones.
rust_test(
    name = "emit_poulpy_instantiate_test",
    srcs = ["emit_poulpy_instantiate_test.rs"],
    deps = [
        ":emit_poulpy_compiles",
        "@crates//:poulpy-cpu-portable",
    ],
)
