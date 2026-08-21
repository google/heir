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

def Lattigo_CKKSBootstrappingEvaluationKeys : Lattigo_CKKSType<"BootstrappingEvaluationKeys", "bootstrapping_eval_keys"> {
  let description = [{
    This type represents the eval keys for bootstrapping for the CKKS encryption scheme.
  }];
  let asmName = "bootstrapping_eval_keys";
}


def Lattigo_CKKSBootstrappingParameter : Lattigo_CKKSType<"BootstrappingParameter", "bootstrapping_parameter"> {
  let description = [{
    This type represents the bootstrapping parameters for the CKKS encryption scheme.
  }];
  let asmName = "bootstrapping_param";
}

def Lattigo_CKKSEvaluator : Lattigo_CKKSType<"Evaluator", "evaluator"> {
  let description = [{
    This type represents the evaluator for the CKKS encryption scheme.
  }];
  let asmName = "evaluator";
}

def Lattigo_CKKSBootstrappingEvaluator : Lattigo_CKKSType<"BootstrappingEvaluator", "bootstrapping_evaluator"> {
  let description = [{
    This type represents the bootstrapping evaluator for the CKKS encryption scheme.
  }];
  let asmName = "bootstrapping_evaluator";
}

def Lattigo_CKKSEncoder : Lattigo_CKKSType<"Encoder", "encoder"> {
  let description = [{
    This type represents the encoder for the CKKS encryption scheme.
  }];
  let asmName = "encoder";
}

def Lattigo_CKKSLinearTransformation : Lattigo_CKKSType<"LinearTransformation", "linear_transformation"> {
  let description = [{
    A prepared (encoded) linear transformation: lattigo's
    lintrans.LinearTransformation, produced by
    lattigo.ckks.prepare_linear_transform and evaluated by
    lattigo.ckks.apply_linear_transform.
  }];
  let asmName = "linear_transformation";
}

def Lattigo_CKKSPolynomialEvaluator : Lattigo_CKKSType<"PolynomialEvaluator", "polynomial_evaluator"> {
  let description = [{
    This type represents the PolynomialEvaluator for the CKKS encryption scheme.
  }];
  let asmName = "poly_evaluator";
}


#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSTYPES_TD_
