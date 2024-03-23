#ifndef INCLUDE_DIALECT_RNS_IR_RNSTYPES_TD_
#define INCLUDE_DIALECT_RNS_IR_RNSTYPES_TD_

include "include/Dialect/RNS/IR/RNSDialect.td"


include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class RNS_Type<string name, string typeMnemonic>
    : TypeDef<RNS_Dialect, name> {
  let mnemonic = typeMnemonic;
}


#endif  // INCLUDE_DIALECT_RNS_IR_RNSTYPES_TD_
