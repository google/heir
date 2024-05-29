""" HEIR bazel workspace """

workspace(name = "heir")

load(
    "@bazel_tools//tools/build_defs/repo:git.bzl",
    "git_repository",
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

# install dependencies for yosys/ABC circuit optimizers
http_archive(
    name = "rules_hdl",
    # Commit 2024-02-14, after merging our patch to fix MacOS builds
    strip_prefix = "bazel_rules_hdl-ef9cce1b82fedb98b56adae1f999885143f5796f",
    url = "https://github.com/hdl/bazel_rules_hdl/archive/ef9cce1b82fedb98b56adae1f999885143f5796f.tar.gz",
)

load("@rules_hdl//dependency_support/at_clifford_yosys:at_clifford_yosys.bzl", "at_clifford_yosys")
load("@rules_hdl//dependency_support/com_github_westes_flex:com_github_westes_flex.bzl", "com_github_westes_flex")
load("@rules_hdl//dependency_support/edu_berkeley_abc:edu_berkeley_abc.bzl", "edu_berkeley_abc")
load("@rules_hdl//dependency_support/net_invisible_island_ncurses:net_invisible_island_ncurses.bzl", "net_invisible_island_ncurses")
load("@rules_hdl//dependency_support/net_zlib:net_zlib.bzl", "net_zlib")
load("@rules_hdl//dependency_support/org_gnu_bison:org_gnu_bison.bzl", "org_gnu_bison")
load("@rules_hdl//dependency_support/org_gnu_gnulib:org_gnu_gnulib.bzl", "org_gnu_gnulib")
load("@rules_hdl//dependency_support/org_gnu_m4:org_gnu_m4.bzl", "org_gnu_m4")
load("@rules_hdl//dependency_support/org_gnu_readline:org_gnu_readline.bzl", "org_gnu_readline")
load("@rules_hdl//dependency_support/org_sourceware_libffi:org_sourceware_libffi.bzl", "org_sourceware_libffi")
load("@rules_hdl//dependency_support/tk_tcl:tk_tcl.bzl", "tk_tcl")

net_invisible_island_ncurses()

org_gnu_readline()

edu_berkeley_abc()

org_gnu_m4()

com_github_westes_flex()

org_gnu_bison()

tk_tcl()

net_zlib()

org_gnu_gnulib()

org_sourceware_libffi()

at_clifford_yosys()

##### Deps for or-tools #####

## Bazel rules.
git_repository(
    name = "platforms",
    commit = "380c85cc2c7b126c6e354f517dc16d89fe760c9f",
    remote = "https://github.com/bazelbuild/platforms.git",
)

git_repository(
    name = "rules_proto",
    commit = "3f1ab99b718e3e7dd86ebdc49c580aa6a126b1cd",
    remote = "https://github.com/bazelbuild/rules_proto.git",
)

## ZLIB
# Would be nice to use llvm-zlib instead here.
new_git_repository(
    name = "zlib",
    build_file = "@com_google_protobuf//:third_party/zlib.BUILD",
    commit = "04f42ceca40f73e2978b50e93806c2a18c1281fc",
    remote = "https://github.com/madler/zlib.git",
)

## Re2
git_repository(
    name = "com_google_re2",
    remote = "https://github.com/google/re2.git",
    tag = "2023-07-01",
)

## Abseil-cpp
git_repository(
    name = "com_google_absl",
    commit = "c2435f8342c2d0ed8101cb43adfd605fdc52dca2",
    patch_args = ["-p1"],
    patches = ["@com_google_ortools//patches:abseil-cpp-20230125.3.patch"],
    remote = "https://github.com/abseil/abseil-cpp.git",
)

## Protobuf
git_repository(
    name = "com_google_protobuf",
    # there's a patch for the CMake build in protobuf, ignoring
    # patches = ["@com_google_ortools//patches:protobuf-v23.3.patch"],
    commit = "4dd15db6eb3955745f379d28fb4a2fcfb6753de3",
    patch_args = ["-p1"],
    remote = "https://github.com/protocolbuffers/protobuf.git",
)

# Load common dependencies.
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

## Solvers
http_archive(
    name = "glpk",
    build_file = "@com_google_ortools//bazel:glpk.BUILD",
    sha256 = "4a1013eebb50f728fc601bdd833b0b2870333c3b3e5a816eeba921d95bec6f15",
    url = "http://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz",
)

http_archive(
    name = "bliss",
    build_file = "@com_google_ortools//bazel:bliss.BUILD",
    patches = ["@com_google_ortools//bazel:bliss-0.73.patch"],
    sha256 = "f57bf32804140cad58b1240b804e0dbd68f7e6bf67eba8e0c0fa3a62fd7f0f84",
    url = "https://github.com/google/or-tools/releases/download/v9.0/bliss-0.73.zip",
    #url = "http://www.tcs.hut.fi/Software/bliss/bliss-0.73.zip",
)

new_git_repository(
    name = "scip",
    build_file = "@com_google_ortools//bazel:scip.BUILD",
    commit = "62fab8a2e3708f3452fad473a6f48715c367316b",
    patch_args = ["-p1"],
    patches = ["@com_google_ortools//bazel:scip.patch"],
    remote = "https://github.com/scipopt/scip.git",
)

# Eigen has no Bazel build.
# Eigen provides a general linear system solver
new_git_repository(
    name = "eigen",
    build_file_content =
        """
cc_library(
    name = 'eigen3',
    srcs = [],
    includes = ['.'],
    hdrs = glob(['Eigen/**']),
    visibility = ['//visibility:public'],
)
""",
    commit = "3147391d946bb4b6c68edd901f2add6ac1f31f8c",
    remote = "https://gitlab.com/libeigen/eigen.git",
)

git_repository(
    name = "highs",
    branch = "bazel",
    remote = "https://github.com/ERGO-Code/HiGHS.git",
)

# ## Swig support
# # pcre source code repository
# new_git_repository(
#     name = "pcre2",
#     build_file = "@com_google_ortools//bazel:pcre2.BUILD",
#     remote = "https://github.com/PCRE2Project/pcre2.git",
#     tag = "pcre2-10.42",
# )

git_repository(
    name = "com_google_ortools",
    commit = "1d696f9108a0ebfd99feb73b9211e2f5a6b0812b",
    remote = "https://github.com/google/or-tools.git",
    shallow_since = "1647023481 +0100",
)

# OpenFHE backend and dependencies
git_repository(
    name = "cereal",
    build_file = "//bazel/openfhe:cereal.BUILD",
    commit = "ebef1e929807629befafbb2918ea1a08c7194554",
    remote = "https://github.com/USCiLab/cereal.git",
)

git_repository(
    name = "rapidjson",
    build_file = "//bazel/openfhe:rapidjson.BUILD",
    commit = "f54b0e47a08782a6131cc3d60f94d038fa6e0a51",
    remote = "https://github.com/Tencent/rapidjson.git",
)

git_repository(
    name = "openfhe",
    build_file = "//bazel/openfhe:openfhe.BUILD",
    # Currently v1.1.4, 2024-03-08
    commit = "94fd76a1d965cfde13f2a540d78ce64146fc2700",
    patches = ["@heir//bazel/openfhe:add_config_core.patch"],
    remote = "https://github.com/openfheorg/openfhe-development.git",
)
