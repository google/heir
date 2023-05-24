"""Configure LLVM Bazel overlays from a 'raw' imported llvm repository"""

# This file was borrowed from tensorflow/third_party/llvm/setup.bzl

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")

# The subset of LLVM targets that HEIR cares about.
_LLVM_TARGETS = [
    "X86",
]

def setup_llvm(name):
    # Build @llvm-project from @llvm-raw using overlays.
    llvm_configure(
        name = name,
        targets = _LLVM_TARGETS,
    )
