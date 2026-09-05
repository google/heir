#ifndef LIB_TRANSFORMS_SOFTMAXTONSSOFTMAX_SOFTMAXTONSSOFTMAX_TD_
#define LIB_TRANSFORMS_SOFTMAXTONSSOFTMAX_SOFTMAXTONSSOFTMAX_TD_

include "mlir/Pass/PassBase.td"

def SoftmaxToNsSoftmax : Pass<"softmax-to-ns-softmax"> {
  let summary = "Lower math_ext.softmax via normalize-and-square";
  let description = [{
    Lowers `math_ext.softmax` to the normalize-and-square scheme
    (Cho et al., CCS'24):

      y = exp((x - M) / 2^k);  then k times:  y <- y^2 / rowsum(y^2)

    The temperature-doubling identity makes the result exactly softmax(x),
    and the iteration is scale-invariant: any uniform per-row error in an
    intermediate normalization cancels in the next round, so only the LAST
    reciprocal's accuracy matters and every reciprocal domain is
    STRUCTURAL — a function of the row length n alone (intermediate sums
    of squares lie in (0, n]; the final sum of squares of a
    probability-like vector lies in [1/n, ~1]) — never of the input
    distribution. The shift M only needs to UPPER-BOUND the scores
    (overshoot cancels exactly), and the /2^k compresses exp's input
    domain by 2^k. Together these make the lowering safe for encrypted
    inputs whose exact ranges are unknowable: unlike a plain
    exp-then-divide softmax, no per-input calibration exists anywhere in
    the decomposition.

    Attribute contract (shared with softmax-to-cgf-softmax): the op's
    `domain_lower`/`domain_upper` give the calibrated score range; M is
    taken as `domain_upper` and exp's stamped domain as
    `[(domain_lower - M)/2^k, 0]`. An optional `exp_degree` overrides the
    stamped Chebyshev degree. Reciprocals are emitted as
    `math_ext.recip` with structural domains, lowered later by
    --lower-recip.

    Padding: when the propagate-padding pass has stamped the softmax
    input with `#tensor_ext.padding` (zero-padded), the lowering keeps
    pad entries out of the row statistics entirely: exp destroys the
    zero-padding (exp(pad - M) != 0), so the exp result is masked ONCE by
    the logical-region indicator (zero is absorbing under the remaining
    squares and multiplies); each round's row sum gets +1 pinned on pad
    ROWS (whose masked sums are exactly 0, outside any reciprocal
    domain); and the structural domains use the LOGICAL row length. The
    pad slots of the result are exactly 0.
  }];
  let options = [
    Option<"nsK", "ns-k", "int", /*default=*/"2",
      "Number of normalize-and-square rounds (temperature halvings).">,
    Option<"expDegree", "exp-degree", "int", /*default=*/"9",
      "Default Chebyshev degree stamped on the emitted exp when the op "
      "carries no exp_degree attribute.">
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::math_ext::MathExtDialect"
  ];
}

#endif  // LIB_TRANSFORMS_SOFTMAXTONSSOFTMAX_SOFTMAXTONSSOFTMAX_TD_
