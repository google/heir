"""Module extensions for MLIR Tutorial dependencies."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def _llvm_deps_impl(_):
    """Implementation of the llvm_deps module extension."""
    LLVM_COMMIT = "ab547095ead5464dc024d66264d9b8a987f429f3"

    # Download LLVM/MLIR using a git repository
    new_git_repository(
        name = "llvm-raw",
        build_file_content = "# empty",
        commit = LLVM_COMMIT,
        init_submodules = False,
        remote = "https://github.com/llvm/llvm-project.git",
        patches = [
            # This patch file contains changes that are fixed in upstream LLVM
            # that are (usually) required to build HEIR, but are not included
            # as of the LLVM_COMMIT hash above (the fixes are still progressing
            # through the automated integration process). The patch file is
            # automatically generated, and should not be removed even if empty.
            "@heir//patches:llvm.patch",
        ],
        patch_args = ["-p1"],
    )

llvm_deps = module_extension(
    implementation = _llvm_deps_impl,
)

CHEDDAR_COMMIT = "307b49cbe03e7f8f14bf31485f716c1090c9ec9d"

def _cheddar_deps_impl(_):
    maybe(
        new_git_repository,
        name = "cheddar",
        build_file = "@heir//bazel/cheddar:cheddar.BUILD",
        commit = CHEDDAR_COMMIT,
        remote = "https://github.com/scale-snu/cheddar-fhe.git",
        patches = ["@heir//patches:cheddar.patch"],
        patch_args = ["-p1"],
    )
    maybe(
        http_archive,
        name = "rmm",
        build_file = "@heir//bazel/cheddar:rmm.BUILD",
        integrity = "sha256-XrU9m0N9g6ABhp9b720nKx1Q2bUk3xzugzvJG3V29ls=",
        strip_prefix = "rmm-22.12.00",
        urls = ["https://github.com/rapidsai/rmm/archive/refs/tags/v22.12.00.tar.gz"],
    )
    maybe(
        http_archive,
        name = "libtommath",
        build_file = "@heir//bazel/cheddar:libtommath.BUILD",
        integrity = "sha256-Bora9RVdKNSsl265XqDfHss2LyDXdyhxVMIqJP2zX6o=",
        strip_prefix = "libtommath-1.2.1",
        urls = ["https://github.com/libtom/libtommath/archive/refs/tags/v1.2.1.tar.gz"],
    )

cheddar_deps = module_extension(implementation = _cheddar_deps_impl)
