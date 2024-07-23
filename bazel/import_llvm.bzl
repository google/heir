"""Provides the repository macro to import LLVM."""

load(
    "@bazel_tools//tools/build_defs/repo:git.bzl",
    "new_git_repository",
)

def import_llvm(name):
    """Imports LLVM."""
    LLVM_COMMIT = "acc159aea1e641e3694ab8fe5faa231788077011"

    new_git_repository(
        name = name,
        # this BUILD file is intentionally empty, because the LLVM project
        # internally contains a set of bazel BUILD files overlaying the project.
        build_file_content = "# empty",
        # this patch should be removed once https://github.com/llvm/llvm-project/commit/0309709a6786653da7164334c83b09c9f37b943a is merged
        patches = ["@heir//bazel:llvm.patch"],
        patch_args = ["-p1"],
        commit = LLVM_COMMIT,
        init_submodules = False,
        remote = "https://github.com/llvm/llvm-project.git",
    )
