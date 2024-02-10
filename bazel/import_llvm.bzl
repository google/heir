"""Provides the repository macro to import LLVM."""

load(
    "@bazel_tools//tools/build_defs/repo:git.bzl",
    "new_git_repository",
)

def import_llvm(name):
    """Imports LLVM."""
    LLVM_COMMIT = "13c14ad42c65e154dc079332dd5dd58e8925d26c"

    new_git_repository(
        name = name,
        # this BUILD file is intentionally empty, because the LLVM project
        # internally contains a set of bazel BUILD files overlaying the project.
        build_file_content = "# empty",
        branch = "fix_mlir_bazel_overlay",
        init_submodules = False,
        remote = "https://github.com/makslevental/llvm-project.git",
    )
