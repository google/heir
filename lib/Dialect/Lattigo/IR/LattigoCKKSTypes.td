#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSTYPES_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSTYPES_TD_

include "LattigoTypes.td"

class Lattigo_CKKSType<string name, string typeMnemonic>
    : Lattigo_Type<"CKKS" # name, "ckks." # typeMnemonic> {
}

def Lattigo_CKKSParameter : Lattigo_CKKSType<"Parameter", "parameter"> {
  let description = [{
    This type represents the parameters for the CKKS encryption scheme.
  }];
  let asmName = "param";
}

def Lattigo_CKKSEncoder : Lattigo_CKKSType<"Encoder", "encoder"> {
  let summary = "Lattigo CKKS encoder type";
  let description = [{
    This type represents an encoder for the Lattigo CKKS encryption scheme.
  }];
  let asmName = "encoder";
}

def Lattigo_CKKSEvaluator : Lattigo_CKKSType<"Evaluator", "evaluator"> {
  let summary = "Lattigo CKKS evaluator type";
  let description = [{
    This type represents an evaluator for the Lattigo CKKS encryption scheme.
  }];
  let asmName = "evaluator";
}

def Lattigo_CKKSRotatedCiphertextMap : Lattigo_CKKSType<"RotatedCiphertextMap", "rotated_ct_map"> {
  let summary = "A map from rotation amount to rotated ciphertext.";
  let description = [{
    A map from rotation amount to rotated ciphertext.
  }];
  let asmName = "rotated_ct_map";
}

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSTYPES_TD_
