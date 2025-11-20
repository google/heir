#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSTYPES_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSTYPES_TD_

include "LattigoAttributes.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

class Lattigo_CKKSType<string name, string typeMnemonic>
    : Lattigo_Type<"CKKS" # name, "ckks." # typeMnemonic> {
}

def Lattigo_CKKSParameter : Lattigo_CKKSType<"Parameter", "parameter"> {
  let description = [{
    This type represents the parameters for the CKKS encryption scheme.
  }];
  let asmName = "param";
}

def Lattigo_CKKSEvaluator : Lattigo_CKKSType<"Evaluator", "evaluator"> {
  let description = [{
    This type represents the evaluator for the CKKS encryption scheme.
  }];
  let asmName = "evaluator";
}

def Lattigo_CKKSEncoder : Lattigo_CKKSType<"Encoder", "encoder"> {
  let description = [{
    This type represents the encoder for the CKKS encryption scheme.
  }];
  let asmName = "encoder";
}

def Lattigo_CKKSPolynomialEvaluator : Lattigo_CKKSType<"PolynomialEvaluator", "polynomial_evaluator"> {
  let description = [{
    This type represents the PolynomialEvaluator for the CKKS encryption scheme.
  }];
  let asmName = "poly_evaluator";
}


#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSTYPES_TD_
