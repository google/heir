#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSTYPES_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSTYPES_TD_

include "LattigoAttributes.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

class Lattigo_CKKSType<string name, string typeMnemonic>
    : Lattigo_Type<"CKKS" # name, "ckks." # typeMnemonic> {
}

// CKKSParameter type definition
def Lattigo_CKKSParameter : Lattigo_CKKSType<"Parameter", "parameter"> {
  let description = [{
    This type represents the parameters for the CKKS encryption scheme.
  }];
  let nameSuggestion = "param";
}

// CKKSEvaluator type definition
def Lattigo_CKKSEvaluator : Lattigo_CKKSType<"Evaluator", "evaluator"> {
  let description = [{
    This type represents the evaluator for the CKKS encryption scheme.
  }];
  let nameSuggestion = "evaluator";
}

// CKKSEncoder type definition
def Lattigo_CKKSEncoder : Lattigo_CKKSType<"Encoder", "encoder"> {
  let description = [{
    This type represents the encoder for the CKKS encryption scheme.
  }];
  let nameSuggestion = "encoder";
}



#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSTYPES_TD_
