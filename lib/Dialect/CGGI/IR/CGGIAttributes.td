#ifndef LIB_DIALECT_CGGI_IR_CGGIATTRIBUTES_TD_
#define LIB_DIALECT_CGGI_IR_CGGIATTRIBUTES_TD_

include "lib/Dialect/CGGI/IR/CGGIDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/CommonAttrConstraints.td"


class CGGI_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CGGI_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

def CGGI_CGGIBoolGates : CGGI_Attr<"CGGIBoolGates", "cggi_bool_gates"> {
  let summary = "An attribute containing an array of strings to store bool gates";

  let description = [{
         This attributes stores a list of integer identifiers for Boolean gates.

     Uses following mapping:
     AND_GATE = 0;
     NAND_GATE = 1;
     OR_GATE = 2;
     NOR_GATE = 3;
     XOR_GATE = 4;
     XNOR_GATE = 5;
     NOT_GATE = 6;
  }];

  let parameters = (ins
      ArrayRefParameter<"::mlir::heir::cggi::CGGIBoolGateEnumAttr">: $gates);

  let assemblyFormat = "`<` $gates `>`";
}

#endif  // LIB_DIALECT_CGGI_IR_CGGIATTRIBUTES_TD_
