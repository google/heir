#ifndef LIB_DIALECT_LWE_IR_LWETRAITS_TD_
#define LIB_DIALECT_LWE_IR_LWETRAITS_TD_

include "mlir/IR/OpBase.td"

def SameOperandsAndResultRings:  NativeOpTrait<"SameOperandsAndResultRings"> {
  let cppNamespace = "::mlir::heir::lwe";
}

#endif // LIB_DIALECT_LWE_IR_LWETRAITS_TD_
