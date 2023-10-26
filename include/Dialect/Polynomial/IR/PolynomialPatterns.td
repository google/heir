#ifndef INCLUDE_DIALECT_POLYNOMIAL_IR_POLYNOMIALPATTERNS_TD_
#define INCLUDE_DIALECT_POLYNOMIAL_IR_POLYNOMIALPATTERNS_TD_

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

#endif  // INCLUDE_DIALECT_POLYNOMIAL_IR_POLYNOMIALPATTERNS_TD_
