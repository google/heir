"""Configure LLVM Bazel overlays from a 'raw' imported llvm repository"""

# This file was borrowed from tensorflow/third_party/llvm/setup.bzl

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")

# The subset of LLVM targets that HEIR cares about.
_LLVM_TARGETS = [
    "AArch64",
    "X86",
    # The bazel dependency graph for mlir-opt fails to load (at the analysis step) without the NVPTX
    # target in this list, because mlir/test:TestGPU depends on the //llvm:NVPTXCodeGen target,
    # which is not defined unless this is included. jkun@ asked the llvm maintiners for tips on how
    # to fix this, and if that yields fruit this can be removed when the fix is upstreamed:
    # https://discord.com/channels/636084430946959380/636732535434510338/1110974700978446368
    "NVPTX",
]

def setup_llvm(name):
    # Build @llvm-project from @llvm-raw using overlays.
    llvm_configure(
        name = name,
        targets = _LLVM_TARGETS,
    )
