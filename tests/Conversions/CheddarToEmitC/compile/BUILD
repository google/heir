load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Compile-only guard that `cheddar-to-emitc` emits C++ which honours CHEDDAR's
# move/const contract. Unlike the GPU end-to-end tests under
# tests/Examples/cheddar, this compiles the emitted code against a header-only
# stub (cheddar_stub.h) with no CUDA, so it runs in normal CI. The cheddar
# dialect and pass are registered unconditionally in heir-opt/-translate, so
# this needs neither a GPU nor --//:enable_cheddar.

heir_opt(
    name = "kernels_emitc",
    src = "kernels.mlir",
    generated_filename = "kernels_emitc.mlir",
    pass_flags = [
        "--cheddar-to-emitc",
        "--reconcile-unrealized-casts",
    ],
)

heir_translate(
    name = "kernels_cpp_raw",
    src = ":kernels_emitc.mlir",
    generated_filename = "kernels_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

# Prepend the includes/usings the emitted body relies on.
genrule(
    name = "kernels_lib_src",
    srcs = [":kernels_raw.cc"],
    outs = ["kernels_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Conversions/CheddarToEmitC/compile/BUILD.
#include <array>
#include <cstdint>
#include "tests/Conversions/CheddarToEmitC/compile/cheddar_stub.h"
using namespace cheddar;
using word = uint64_t;
PRELUDE_EOF
cat $(location :kernels_raw.cc) >> $@
""",
)

cc_library(
    name = "cheddar_stub",
    hdrs = ["cheddar_stub.h"],
)

cc_library(
    name = "kernels_compiled",
    srcs = [":kernels_lib_src"],
    deps = [":cheddar_stub"],
)

# Compiling the generated C++ against the stub *is* the assertion: if the
# emitter produced C++ that violates CHEDDAR's move/const contract, this fails
# to build. build_test compiles without linking, so the stub only needs
# declarations (no method bodies), and no GPU/CUDA is involved.
build_test(
    name = "compile_test",
    targets = [":kernels_compiled"],
)
