#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALPATTERNS_TD_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALPATTERNS_TD_

include "PolynomialOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

// A hack to get the proper integer attribute type from the (possibly
// non-power-of-two) ring coefficient modulus.
// TODO: perhaps we can move this logic into the semantics of mul_scalar,
// or else include it in the lowering.
def getPolynomialCoefficientIntAttribute
  : NativeCodeCall<
      "$_builder.getIntegerAttr("
        "IntegerType::get("
          "$1.getContext(), "
          "($1.getRing().coefficientModulus() - 1).getActiveBits(), "
          "IntegerType::SignednessSemantics::Signless"
        "), "
        "$2"
      ")">;

def SubAsAdd : Pat<
  (Polynomial_SubOp $f, $g),
  (Polynomial_AddOp $f,
    (Polynomial_MulScalarOp $g,
      (Arith_ConstantOp
        ConstantAttr<I32Attr, "-1">)))>;
        // FIXME: get this to work
        // (getPolynomialCoefficientIntAttribute $f, -1))))>;

// %0 = _polynomial.ntt %p
// %1 = _polynomial.intt %0
// ---> %p
def INTTAfterNTT : Pat<
  (Polynomial_INTTOp
    (Polynomial_NTTOp $poly)
  ),
  (replaceWithValue $poly),
  []
>;

// %0 = _polynomial.intt %t
// %1 = _polynomial.ntt %0
// ---> %t
def NTTAfterINTT : Pat<
  (Polynomial_NTTOp
    (Polynomial_INTTOp $tensor)
  ),
  (replaceWithValue $tensor),
  []
>;

#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALPATTERNS_TD_
