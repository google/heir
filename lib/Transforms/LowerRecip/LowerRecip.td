#ifndef LIB_TRANSFORMS_LOWERRECIP_LOWERRECIP_TD_
#define LIB_TRANSFORMS_LOWERRECIP_LOWERRECIP_TD_

include "mlir/Pass/PassBase.td"

def LowerRecip : Pass<"lower-recip", "ModuleOp"> {
  let summary = "Lower math_ext.recip to Goldschmidt iteration (mult/add only).";
  let description = [{
    Lowers `math_ext.recip` to an FHE-friendly Goldschmidt iteration
    (https://github.com/google/heir/issues/2936): an affine minimax initial
    estimate `x0 = alpha + beta*d` over the op's annotated
    `domain_lower`/`domain_upper` (both required, must be positive), followed
    by `num-iterations` steps of `x_{k+1} = x_k * (2 - d * x_k)`. The whole
    lowering is multiplications, additions, and constants — no division and
    no polynomial-approximation machinery.

    The relative error after the seed is
    `E = (hi-lo)^2 / ((hi+lo)^2 + 4*lo*hi)` and squares each iteration.
    The pass derives the iteration count from the stamped domain: the
    smallest n with `E^(2^n) <= 1e-4`, floored at `num-iterations` (so
    tight domains keep the historical 6-iteration circuits bit-identical)
    and capped at 24. Wide, conservatively stamped domains therefore stay
    accurate automatically instead of silently losing precision — the
    stamped domain only has to COVER the data, not fit it tightly.
    Ops without domain attributes are left untouched (and will fail loudly
    downstream), since a reciprocal without a known positive domain has no
    sound FHE lowering.
  }];
  let options = [
    Option<
      "numIterations",
      "num-iterations",
      "int",
      /*default=*/"6",
      "Minimum number of Goldschmidt refinement iterations after the "
      "affine seed; the pass raises this to whatever the stamped domain "
      "needs for 1e-4 relative error."
    >
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::math_ext::MathExtDialect"
  ];
}

#endif  // LIB_TRANSFORMS_LOWERRECIP_LOWERRECIP_TD_
