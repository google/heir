load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Compiles the emitted C++ against the CHEDDAR library, so an API mismatch or a
# move/const violation fails the build. Compile-only (no GPU needed), but
# requires `--config=cheddar`; see .github/workflows/build_cheddar.yml.

heir_opt(
    name = "kernels_emitc",
    src = "kernels.mlir",
    generated_filename = "kernels_emitc.mlir",
    pass_flags = [
        "--cheddar-bufferize",
        "--fold-memref-alias-ops",
        "--cse",
        "--canonicalize",
        "--drop-equivalent-buffer-results",
        "--buffer-results-to-out-params=hoist-static-allocs=true modify-public-functions=true add-result-attr=true",
        "--canonicalize",
        "--convert-to-emitc=filter-dialects=cheddar,arith,scf",
        "--cheddar-emitc-boundary",
        "--reconcile-unrealized-casts",
    ],
)

heir_translate(
    name = "kernels_cpp",
    src = ":kernels_emitc.mlir",
    generated_filename = "kernels.cc",
    pass_flags = ["--mlir-to-cpp"],
)

cc_library(
    name = "kernels_compiled",
    srcs = [":kernels.cc"],
    target_compatible_with = requires_cheddar(),
    deps = [
        "@cheddar",
    ],
)

build_test(
    name = "compile_test",
    target_compatible_with = requires_cheddar(),
    targets = [":kernels_compiled"],
)
