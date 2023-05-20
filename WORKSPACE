""" HEIR bazel workspace """

workspace(name = "heir")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

# provides the `license` rule, which is required because gentbl_rule implicitly
# depends upon the target '//:license'. How bizarre.
http_archive(
    name = "rules_license",
    sha256 = "6157e1e68378532d0241ecd15d3c45f6e5cfd98fc10846045509fb2a7cc9e381",
    urls = [
        "https://github.com/bazelbuild/rules_license/releases/download/0.0.4/rules_license-0.0.4.tar.gz",
    ],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

# LLVM is pinned to the same commit used in the Google monorepo, and then
# imported into this workspace as a git repository. Then the build files
# defined in the LLVM monorepo are overlaid using llvm_configure in the setup
# script below. This defines the @llvm-project which is used for llvm build
# dependencies.
load("//bazel:import_llvm.bzl", "import_llvm")

import_llvm("llvm-raw")

load("//bazel:setup_llvm.bzl", "setup_llvm")

setup_llvm("llvm-project")

# LLVM doesn't have proper support for excluding this optional dependency,
# but it is supposed to make LLVM faster, so why not include it.
# See https://reviews.llvm.org/D143344#4232172
maybe(
    http_archive,
    name = "llvm_zstd",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
    sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
    strip_prefix = "zstd-1.5.2",
    urls = [
        "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
    ],
)
