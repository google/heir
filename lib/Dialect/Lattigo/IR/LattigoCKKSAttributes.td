#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSATTRIBUTES_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSATTRIBUTES_TD_

class Lattigo_CKKSAttribute<string attrName, string attrMnemonic>
      : Lattigo_Attribute<"CKKS" # attrName, "ckks." # attrMnemonic> {
    let summary = "Attribute for Lattigo CKKS";
    let description = [{
        This attribute represents the values for Lattigo CKKS.
    }];
}

def Lattigo_CKKSParametersLiteral
    : Lattigo_CKKSAttribute<"ParametersLiteral", "parameters_literal"> {
    let summary = "Literal parameters for Lattigo CKKS";
    let description = [{
        This attribute represents the literal parameters for Lattigo CKKS.

        This is in accordance with https://pkg.go.dev/github.com/tuneinsight/lattigo/v6@v6.1.0/schemes/ckks#ParametersLiteral
        where some field are not present in the current implementation.

        Users must set the polynomial degree (LogN) and the coefficient modulus,
        by either setting the Q and P fields to the desired moduli chain,
        or by setting the LogQ and LogP fields to the desired moduli sizes.

        Note that for Lattigo, Q/P requires []uint64, where this attribute
        only provides int64. We assume user should not select moduli so large
        to consider the signedness issue.

        Users must also specify a default initial scale for the plaintexts1.
    }];
    let parameters = (ins
      "int":$logN,
      OptionalParameter<"DenseI64ArrayAttr">:$Q,
      OptionalParameter<"DenseI64ArrayAttr">:$P,
      OptionalParameter<"DenseI32ArrayAttr">:$logQ,
      OptionalParameter<"DenseI32ArrayAttr">:$logP,
      "int":$logDefaultScale
    );
}

def Lattigo_CKKSBootstrappingParametersLiteral
    : Lattigo_CKKSAttribute<"BootstrappingParametersLiteral", "bootstrapping_parameters_literal"> {
    let summary = "Literal bootstrapping parameters for Lattigo CKKS";
    let description = [{
        This attribute represents the literal bootstrapping parameters for Lattigo CKKS.

        This is in accordance with https://pkg.go.dev/github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping#ParametersLiteral
        where some field are not present in the current implementation.

        This struct contains only optional fields.
    }];
    let parameters = (ins
      OptionalParameter<"int">:$logN
    );
}

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOCKKSATTRIBUTES_TD_
