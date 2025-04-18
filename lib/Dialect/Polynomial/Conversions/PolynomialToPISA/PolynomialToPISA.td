#ifndef LIB_DIALECT_POLYNOIAL_CONVERSIONS_POLYNOMIALTOPISA_POLYNOMIALTOPISA_TD_
#define LIB_DIALECT_POLYNOIAL_CONVERSIONS_POLYNOMIALTOPISA_POLYNOMIALTOPISA_TD_

include "mlir/Pass/PassBase.td"

def PolynomialToPISA : Pass<"polynomial-to-pisa"> {
  let summary = "Lower `polynomial` to `pisa` dialect.";

  let description = [{
    "This pass converts `polynomial` (and relevant `arith_ext`) operations to `pisa` operations.
     Note that the polynomials/tensors (polynomial after NTT) must have degree/size 8192.
     This pass does not support polynomial operations on tensors of polynomials,
     nor arith_ext operations on multi-dimensional tensors. (cf. `--convert-elementwise-to-affine`)."
  }];

  let dependentDialects = [
    "mlir::heir::polynomial::PolynomialDialect",
    "mlir::heir::pisa::PISADialect",
  ];
}

#endif  // LIB_DIALECT_POLYNOIAL_CONVERSIONS_POLYNOMIALTOPISA_POLYNOMIALTOPISA_TD_
