#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_TD_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_TD_

include "CheddarDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"

class Cheddar_Attribute<string attrName, string attrMnemonic>
    : AttrDef<Cheddar_Dialect, attrName> {
    let summary = "Attribute for CHEDDAR";
    let description = [{
        This attribute represents configuration values for CHEDDAR.
    }];
    let mnemonic = attrMnemonic;
    let assemblyFormat = "`<` struct(params) `>`";
}

def Cheddar_ParameterFileAttr
    : Cheddar_Attribute<"ParameterFile", "parameter_file"> {
    let summary = "Path to a CHEDDAR parameter JSON file";
    let description = [{
        This attribute holds the path to a CHEDDAR parameter JSON file.

        CHEDDAR parameters are loaded from pre-built JSON files that specify
        ring dimension, prime chains, scale, and other CKKS parameters.
    }];
    let parameters = (ins
      StringRefParameter<"path to JSON parameter file">:$path,
      DefaultValuedParameter<"bool", "false", "use 64-bit word type">:$use64Bit
    );
}

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDARATTRIBUTES_TD_
