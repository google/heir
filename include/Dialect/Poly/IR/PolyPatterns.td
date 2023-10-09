#ifndef INCLUDE_DIALECT_POLY_IR_POLYPATTERNS_TD_
#define INCLUDE_DIALECT_POLY_IR_POLYPATTERNS_TD_

include "PolyOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

// A hack to get the proper integer attribute type from the (possibly
// non-power-of-two) ring coefficient modulus.
// TODO: perhaps we can move this logic into the semantics of mul_scalar,
// or else include it in the lowering.
def getPolyCoefficientIntAttribute
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
  (Poly_SubOp $f, $g),
  (Poly_AddOp $f,
    (Poly_MulConstantOp $g,
      (Arith_ConstantOp
        ConstantAttr<I32Attr, "-1">)))>;
        // FIXME: get this to work
        // (getPolyCoefficientIntAttribute $f, -1))))>;

#endif  // INCLUDE_DIALECT_POLY_IR_POLYPATTERNS_TD_
