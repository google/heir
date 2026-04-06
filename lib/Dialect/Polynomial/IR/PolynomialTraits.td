#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALTRAITS_TD_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALTRAITS_TD_

include "mlir/IR/OpBase.td"

def SameOperandsAndResultForm:  NativeOpTrait<"SameOperandsAndResultForm"> {
  let cppNamespace = "::mlir::heir::polynomial";
}

def FixedFormCoeff:  NativeOpTrait<"FixedFormCoeff"> {
  let cppNamespace = "::mlir::heir::polynomial";
}

def FixedFormEval:  NativeOpTrait<"FixedFormEval"> {
  let cppNamespace = "::mlir::heir::polynomial";
}

#endif // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALTRAITS_TD_
