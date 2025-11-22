"""Provides the repository macro to import LLVM."""

load(
    "@bazel_tools//tools/build_defs/repo:git.bzl",
    "new_git_repository",
)

def import_llvm(name):
    """Imports LLVM."""
    LLVM_COMMIT = "d65be16ab6adf00af21e75d29049ae5de0f3a38a"

    new_git_repository(
        name = name,
        # this BUILD file is intentionally empty, because the LLVM project
        # internally contains a set of bazel BUILD files overlaying the project.
        build_file_content = "# empty",
        patches = [
            # This patch file contains changes that are fixed in upstream LLVM
            # that are (usually) required to build HEIR, but are not included
            # as of the LLVM_COMMIT hash above (the fixes are still progressing
            # through the automated integration process). The patch file is
            # automatically generated, and should not be removed even if empty.
            "@heir//patches:llvm.patch",
        ],
        patch_args = ["-p1"],
        commit = LLVM_COMMIT,
        init_submodules = False,
        remote = "https://github.com/llvm/llvm-project.git",
    )
