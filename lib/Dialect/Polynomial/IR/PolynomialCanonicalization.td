#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALCANONICALIZATION_TD_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALCANONICALIZATION_TD_

include "lib/Dialect/Polynomial/IR/PolynomialOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

def Equal : Constraint<CPred<"$0 == $1">>;

def INTTAfterNTT : Pat<
  (Polynomial_INTTOp (Polynomial_NTTOp $poly, $r1), $r2),
  (replaceWithValue $poly),
  [(Equal $r1, $r2)]
>;

def NTTAfterINTT : Pat<
  (Polynomial_NTTOp (Polynomial_INTTOp $tensor, $r1), $r2),
  (replaceWithValue $tensor),
  [(Equal $r1, $r2)]
>;

#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALCANONICALIZATION_TD_
