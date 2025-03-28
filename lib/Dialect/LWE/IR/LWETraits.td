#ifndef LIB_DIALECT_LWE_IR_LWETRAITS_TD_
#define LIB_DIALECT_LWE_IR_LWETRAITS_TD_

include "mlir/IR/OpBase.td"

def SameOperandsAndResultRings:  NativeOpTrait<"SameOperandsAndResultRings"> {
  let cppNamespace = "::mlir::heir::lwe";
}

def SameOperandsAndResultPlaintextTypes:  NativeOpTrait<"SameOperandsAndResultPlaintextTypes"> {
  let cppNamespace = "::mlir::heir::lwe";
}

def AllCiphertextTypesMatch:  NativeOpTrait<"AllCiphertextTypesMatch"> {
  let cppNamespace = "::mlir::heir::lwe";
}

def IsCiphertextPlaintextOp:  NativeOpTrait<"IsCiphertextPlaintextOp"> {
  let cppNamespace = "::mlir::heir::lwe";
}

#endif // LIB_DIALECT_LWE_IR_LWETRAITS_TD_
