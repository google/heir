#ifndef LIB_TRANSFORMS_STAMPAPPROXIMATIONDOMAINS_STAMPAPPROXIMATIONDOMAINS_TD_
#define LIB_TRANSFORMS_STAMPAPPROXIMATIONDOMAINS_STAMPAPPROXIMATIONDOMAINS_TD_

include "mlir/Pass/PassBase.td"

def StampApproximationDomains : Pass<"stamp-approximation-domains", "ModuleOp"> {
  let summary = "Stamp calibrated approximation domains onto unannotated math ops";
  let description = [{
    Stamps `degree`/`domain_lower`/`domain_upper` onto `math.rsqrt` and
    `math.erf` ops that carry none. torch exports emit these ops bare
    (LayerNorm rsqrt, GELU erf), and the polynomial-approximation pass's
    generic default domain of [-1, 1] misfits both badly: rsqrt's argument
    is a small positive variance range (measured max_abs_err ~2.0 on a real
    LayerNorm with the default vs a calibrated domain), and erf's argument
    spans the pre-activation range (measured +-8.66 on bert-tiny; outside
    [-1, 1] the default approximation returns garbage).

    The stamped values are calibration policy, not structure: defaults
    cover bert-tiny's calibrated ranges and are overridable per model via
    the pass options. Ops that already carry a domain_lower attribute are
    left untouched, so a frontend that stamps its own calibration wins.
  }];
  let options = [
    Option<"rsqrtDegree", "rsqrt-degree", "int", /*default=*/"12",
      "Chebyshev degree stamped on unannotated math.rsqrt ops.">,
    Option<"rsqrtDomainLower", "rsqrt-domain-lower", "double",
      /*default=*/"0.5",
      "domain_lower stamped on unannotated math.rsqrt ops.">,
    Option<"rsqrtDomainUpper", "rsqrt-domain-upper", "double",
      /*default=*/"5.0",
      "domain_upper stamped on unannotated math.rsqrt ops.">,
    Option<"erfDegree", "erf-degree", "int", /*default=*/"27",
      "Chebyshev degree stamped on unannotated math.erf ops.">,
    Option<"erfDomainBound", "erf-domain-bound", "double",
      /*default=*/"9.0",
      "Symmetric domain bound stamped on unannotated math.erf ops "
      "(covers bert-tiny's calibrated GELU input range of +-8.66; the "
      "generic [-1, 1] default returns garbage outside it).">
  ];
  let dependentDialects = [
    "mlir::math::MathDialect"
  ];
}

#endif  // LIB_TRANSFORMS_STAMPAPPROXIMATIONDOMAINS_STAMPAPPROXIMATIONDOMAINS_TD_
