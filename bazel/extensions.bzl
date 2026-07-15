"""Module extensions for MLIR Tutorial dependencies."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def _llvm_deps_impl(_):
    """Implementation of the llvm_deps module extension."""
    LLVM_COMMIT = "6e6272088e69834fde6e30f0033f36ee63806d00"

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
