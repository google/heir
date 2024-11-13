#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALCANONICALIZATION_TD_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALCANONICALIZATION_TD_

include "lib/Dialect/Polynomial/IR/PolynomialOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

def Equal : Constraint<CPred<"$0 == $1">>;

// Get a -1 integer attribute of the same type as the polynomial SSA value's
// ring coefficient type.
def getMinusOne
  : NativeCodeCall<
      "$_builder.getIntegerAttr("
        "cast<PolynomialType>($0.getType()).getRing().getCoefficientType(), -1)">;

def SubAsAdd : Pat<
  (Polynomial_SubOp $f, $g),
  (Polynomial_AddOp $f,
    (Polynomial_MulScalarOp $g,
      (Arith_ConstantOp (getMinusOne $g))))>;

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
