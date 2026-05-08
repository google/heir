#ifndef LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDATTRIBUTES_TD_
#define LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDATTRIBUTES_TD_

include "JaxiteWordDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

class JaxiteWord_Attribute<string attrName, string attrMnemonic>
    : AttrDef<JaxiteWord_Dialect, attrName> {
    let mnemonic = attrMnemonic;
    let assemblyFormat = "`<` struct(params) `>`";
}

def JaxiteWord_CkksParameters : JaxiteWord_Attribute<"CkksParameters", "ckks_parameters"> {
  let summary = "Jaxite CKKS parameters";
  let description = [{
    Parameters for Jaxite CKKS backend.
  }];

  let parameters = (ins
    "DenseI64ArrayAttr":$q_towers,
    "DenseI64ArrayAttr":$p_towers,
    "int":$r,
    "int":$c,
    "int":$dnum,
    "int":$composite_degree,
    "int":$batch
  );
}

#endif // LIB_DIALECT_JAXITEWORD_IR_JAXITEWORDATTRIBUTES_TD_
