#ifndef LIB_DIALECT_ORION_IR_ORIONOPS_TD_
#define LIB_DIALECT_ORION_IR_ORIONOPS_TD_

include "lib/Dialect/Orion/IR/OrionDialect.td"

include "mlir/IR/OpBase.td"

class Orion_Op<string mnemonic, list<Trait> traits = []> :
        Op<Orion_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::orion";
}

#endif  // LIB_DIALECT_ORION_IR_ORIONOPS_TD_
