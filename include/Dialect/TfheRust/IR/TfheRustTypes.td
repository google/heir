#ifndef INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTTYPES_TD_
#define INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTTYPES_TD_

include "TfheRustDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

// A base class for all types in this dialect
class TfheRust_Type<string name, string typeMnemonic>
    : TypeDef<TfheRust_Dialect, name> {
  let mnemonic = typeMnemonic;
}

#endif  // INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTTYPES_TD_
