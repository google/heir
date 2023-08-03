""" HEIR bazel workspace """

workspace(name = "heir")

load(
    "@bazel_tools//tools/build_defs/repo:git.bzl",
    "new_git_repository",
)
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

# LLVM doesn't have proper support for excluding the optional llvm_zstd and
# llvm_zlib dependencies but it is supposed to make LLVM faster, so why not
# include it.
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

maybe(
    http_archive,
    name = "llvm_zlib",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
    sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
    strip_prefix = "zlib-ng-2.0.7",
    urls = [
        "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
    ],
)

# rules_foreign_cc provides access to a `make` bazel rule, which is needed
# to build yosys
http_archive(
    name = "rules_foreign_cc",
    sha256 = "bcd0c5f46a49b85b384906daae41d277b3dc0ff27c7c752cc51e43048a58ec83",
    strip_prefix = "rules_foreign_cc-0.7.1",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.7.1.tar.gz",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

# Add a yosys dependency for circuit optimization and simulation
# in the heir-translate --export_verilog path
new_git_repository(
    name = "yosys",
    build_file = "//bazel:yosys.BUILD",
    # As of 2023-04-24
    commit = "cee3cb31b98e3b67af3165969c8cfc0616c37e19",
    remote = "https://github.com/YosysHQ/yosys.git",
)

# For non-LIT unit testing
http_archive(
    name = "googletest",
    sha256 = "730215d76eace9dd49bf74ce044e8daa065d175f1ac891cc1d6bb184ef94e565",
    strip_prefix = "googletest-f53219cdcb7b084ef57414efea92ee5b71989558",
    urls = [
        "https://github.com/google/googletest/archive/f53219cdcb7b084ef57414efea92ee5b71989558.tar.gz",  # 2023-03-16
    ],
)

# googletest comes with abseil as @com_google_absl, see
# https://github.com/google/googletest/blob/23f642ab2317c632d93326c65efd44671c1d9985/googletest_deps.bzl
load("@googletest//:googletest_deps.bzl", "googletest_deps")

googletest_deps()

# Depend on a hermetic python version
new_git_repository(
    name = "rules_python",
    commit = "9ffb1ecd9b4e46d2a0bca838ac80d7128a352f9f",  # v0.23.1
    remote = "https://github.com/bazelbuild/rules_python.git",
)

load("@rules_python//python:repositories.bzl", "python_register_toolchains")

python_register_toolchains(
    name = "python3_10",
    # Available versions are listed at
    # https://github.com/bazelbuild/rules_python/blob/main/python/versions.bzl
    python_version = "3.10",
)

load("@python3_10//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "heir_pip_deps",
    python_interpreter_target = interpreter,
    requirements_lock = "//:requirements.txt",
)

load("@heir_pip_deps//:requirements.bzl", "install_deps")

install_deps()

# compile_commands extracts the relevant compile data from bazel into
# `compile_commands.json` so that clangd, clang-tidy, etc., can use it.
# Whenever a build file changes, you must re-run
#
#   bazel run @hedron_compile_commands//:refresh_all
#
# to ingest new data into these tools.
#
# See the project repo for more details and configuration options
# https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",
    sha256 = "3cd0e49f0f4a6d406c1d74b53b7616f5e24f5fd319eafc1bf8eee6e14124d115",
    strip_prefix = "bazel-compile-commands-extractor-3dddf205a1f5cde20faf2444c1757abe0564ff4c",
    # 2023-05-12
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/3dddf205a1f5cde20faf2444c1757abe0564ff4c.tar.gz",
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

hedron_compile_commands_setup()
