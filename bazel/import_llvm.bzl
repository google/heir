"""Provides the repository macro to import LLVM."""

load(
    "@bazel_tools//tools/build_defs/repo:git.bzl",
    "new_git_repository",
)

def import_llvm(name):
    """Imports LLVM."""
    LLVM_COMMIT = "69d0bab82689d470e3fd68f50ca8b8d28f3e2294"

    new_git_repository(
        name = name,
        # this BUILD file is intentionally empty, because the LLVM project
        # internally contains a set of bazel BUILD files overlaying the project.
        build_file_content = "# empty",
        # this patch should be removed once llvm/llvm-project/pull/115188 is merged
        patches = ["@heir//bazel:llvm.patch"],
        patch_args = ["-p1"],
        commit = LLVM_COMMIT,
        init_submodules = False,
        remote = "https://github.com/llvm/llvm-project.git",
    )
