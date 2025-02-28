"""Provides the repository macro to import LLVM."""

load(
    "@bazel_tools//tools/build_defs/repo:git.bzl",
    "new_git_repository",
)

def import_llvm(name):
    """Imports LLVM."""
    LLVM_COMMIT = "9e2eb95c238d5d7b059da766b24e5a01c683bf7a"

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
