#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOTYPES_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOTYPES_TD_

include "LattigoDialect.td"
include "LattigoAttributes.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class Lattigo_Type<string name, string typeMnemonic>
    : TypeDef<Lattigo_Dialect, name> {
  let mnemonic = typeMnemonic;
}

include "LattigoBGVTypes.td"
include "LattigoRLWETypes.td"

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOTYPES_TD_
