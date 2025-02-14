#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOATTRIBUTES_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOATTRIBUTES_TD_

include "LattigoDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"

class Lattigo_Attribute<string attrName, string attrMnemonic>
    : AttrDef<Lattigo_Dialect, attrName> {
    let summary = "Attribute for Lattigo";
    let description = [{
        This attribute represents the values for Lattigo.
    }];
    let mnemonic = attrMnemonic;
    let assemblyFormat = "`<` struct(params) `>`";
}

include "LattigoBGVAttributes.td"
include "LattigoCKKSAttributes.td"

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOATTRIBUTES_TD_
