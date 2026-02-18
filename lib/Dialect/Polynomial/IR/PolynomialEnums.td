#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALENUMS_TD_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALENUMS_TD_

include "lib/Dialect/Polynomial/IR/PolynomialDialect.td"
include "mlir/IR/EnumAttr.td"

def Polynomial_Form_Coeff : I32EnumAttrCase<"COEFF", 0, "coeff">;
def Polynomial_Form_Eval  : I32EnumAttrCase<"EVAL", 1, "eval">;

def Polynomial_FormEnum : I32EnumAttr<"Form", "Polynomial representation form",
  [Polynomial_Form_Coeff, Polynomial_Form_Eval]> {
  let cppNamespace = "::mlir::heir::polynomial";
}


#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALENUMS_TD_
