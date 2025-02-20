#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOBGVTYPES_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOBGVTYPES_TD_

include "LattigoAttributes.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

class Lattigo_BGVType<string name, string typeMnemonic>
    : Lattigo_Type<"BGV" # name, "bgv." # typeMnemonic> {
}

// BGVParameter type definition
def Lattigo_BGVParameter : Lattigo_BGVType<"Parameter", "parameter"> {
  let description = [{
    This type represents the parameters for the BGV encryption scheme.
  }];
  let asmName = "param";
}

// BGVEvaluator type definition
def Lattigo_BGVEvaluator : Lattigo_BGVType<"Evaluator", "evaluator"> {
  let description = [{
    This type represents the evaluator for the BGV encryption scheme.
  }];
  let asmName = "evaluator";
}

// BGVEncoder type definition
def Lattigo_BGVEncoder : Lattigo_BGVType<"Encoder", "encoder"> {
  let description = [{
    This type represents the encoder for the BGV encryption scheme.
  }];
  let asmName = "encoder";
}



#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOBGVTYPES_TD_
