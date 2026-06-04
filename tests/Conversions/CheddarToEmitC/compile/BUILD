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
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <vector>
#include "tests/Conversions/CheddarToEmitC/compile/cheddar_stub.h"
using namespace cheddar;
using word = uint64_t;
// HEIR-side emitter shim (see the orion_mlp_lt BUILD for the rationale): the
// cheddar.linear_transform lowering calls this, so the compile guard must see it.
template <int W, typename wordT>
void RunLinearTransform(Ciphertext<wordT>& out, Context<wordT>* ctx,
                        const Ciphertext<wordT>& in,
                        const EvkMap<wordT>& evk_map, double diag[][W],
                        std::initializer_list<int> idx, int level, int bs,
                        int gs) {
  StripedMatrix m(W, W);
  int d = 0;
  for (int k : idx) {
    m[k] = std::vector<Complex>(diag[d], diag[d] + W);
    ++d;
  }
  ConstContextPtr<wordT> cp(ConstContextPtr<wordT>(), ctx);
  LinearTransform<wordT> lt(cp, m, level, ctx->param_.GetScale(level), bs, gs);
  lt.Evaluate(cp, out, in, evk_map);
}
// Same shape for cheddar.eval_poly -> CHEDDAR's EvalPoly class (Chebyshev basis).
template <typename wordT>
void RunEvalPoly(Ciphertext<wordT>& out, Context<wordT>* ctx,
                 const Ciphertext<wordT>& in, const EvkMap<wordT>& evk_map,
                 std::initializer_list<double> coeffs, int level,
                 int output_level) {
  ConstContextPtr<wordT> cp(ConstContextPtr<wordT>(), ctx);
  std::vector<double> c(coeffs);
  EvalPoly<wordT> poly(c, level, ctx->param_.GetScale(level),
                       ctx->param_.GetScale(output_level), /*chebyshev=*/true);
  poly.Compile(cp);
  poly.Evaluate(cp, out, in, evk_map.GetMultiplicationKey());
}
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
