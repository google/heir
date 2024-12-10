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

        This is in accordance with https://pkg.go.dev/github.com/tuneinsight/lattigo/v6@v6.1.0/schemes/bgv#ParametersLiteral
        where some field are not present in the current implementation.

        Users must set the polynomial degree (LogN) and the coefficient modulus,
        by either setting the Q and P fields to the desired moduli chain,
        or by setting the LogQ and LogP fields to the desired moduli sizes.

        Note that for Lattigo, Q/P requires []uint64, where this attribute
        only provides int64. We assume user should not select moduli so large
        to consider the signedness issue.

        Users must also specify the coefficient modulus in plaintext-space (T).
        This modulus must be an NTT-friendly prime in the plaintext space:
        it must be equal to 1 modulo 2n where n is the plaintext ring degree
        (i.e., the plaintext space has n slots).
    }];
    let parameters = (ins
      "int":$logN,
      OptionalParameter<"DenseI64ArrayAttr">:$Q,
      OptionalParameter<"DenseI64ArrayAttr">:$P,
      OptionalParameter<"DenseI32ArrayAttr">:$logQ,
      OptionalParameter<"DenseI32ArrayAttr">:$logP,
      "int64_t":$plaintextModulus
    );
}

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOBGVATTRIBUTES_TD_
