#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOBGVATTRIBUTES_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOBGVATTRIBUTES_TD_

class Lattigo_BGVAttribute<string attrName, string attrMnemonic>
      : Lattigo_Attribute<"BGV" # attrName, "bgv." # attrMnemonic> {
    let summary = "Attribute for Lattigo BGV";
    let description = [{
        This attribute represents the values for Lattigo BGV.
    }];
}

def Lattigo_BGVParametersLiteral
    : Lattigo_BGVAttribute<"ParametersLiteral", "parameters_literal"> {
    let summary = "Literal parameters for Lattigo BGV";
    let description = [{
        This attribute represents the literal parameters for Lattigo BGV.
    }];
    let parameters = (ins
      "int":$logN,
      "DenseI32ArrayAttr":$logQ,
      "DenseI32ArrayAttr":$logP,
      "int64_t":$plaintextModulus
    );
}

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOBGVATTRIBUTES_TD_
