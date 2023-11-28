#ifndef HEIR_INCLUDE_DIALECT_BGV_IR_BGVATTRIBUTES_TD_
#define HEIR_INCLUDE_DIALECT_BGV_IR_BGVATTRIBUTES_TD_

include "BGVDialect.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpBase.td"

def BGVRingArrayAttr : AttrDef<BGV_Dialect, "BGVRings"> {
  let mnemonic = "rings";
  let parameters = (ins ArrayRefParameter<"::mlir::heir::polynomial::RingAttr">:$rings);
  let assemblyFormat = "`<` $rings `>`";
}

#endif  // HEIR_INCLUDE_DIALECT_BGV_IR_BGVATTRIBUTES_TD_
